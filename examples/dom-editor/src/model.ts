// Worker-backed client for the Needle engine.
//
// All inference happens off the main thread (see src/worker.ts). This client
// is a thin postMessage wrapper that hands out promises for each request.

export type LoadProgress = {
  stage: "init" | "weights" | "vocab" | "engine" | "ready";
  loadedBytes?: number;
  totalBytes?: number;
};

type InMsg =
  | { id: number; type: "load" }
  | { id: number; type: "infer"; query: string; toolsJson: string }
  | { id: number; type: "retrieve"; query: string; descriptions: string[]; topK: number }
  | { id: number; type: "hasContrastive" };

type OutMsg =
  | { id: number; type: "progress"; stage: LoadProgress["stage"]; loadedBytes?: number; totalBytes?: number }
  | { id: number; type: "result"; data: unknown }
  | { id: number; type: "error"; message: string };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  onProgress?: (p: LoadProgress) => void;
};

export class NeedleModel {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private contrastiveAvailable = false;

  private constructor(worker: Worker) {
    this.worker = worker;
    this.worker.addEventListener("message", (ev: MessageEvent<OutMsg>) => this.onMessage(ev.data));
    this.worker.addEventListener("error", (ev) => {
      // Reject every outstanding request; without this they'd hang forever.
      const err = new Error(`worker error: ${ev.message}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    });
  }

  private onMessage(msg: OutMsg): void {
    const p = this.pending.get(msg.id);
    if (!p) return;
    if (msg.type === "progress") {
      p.onProgress?.({ stage: msg.stage, loadedBytes: msg.loadedBytes, totalBytes: msg.totalBytes });
      return; // load isn't done yet
    }
    this.pending.delete(msg.id);
    if (msg.type === "result") p.resolve(msg.data);
    else p.reject(new Error(msg.message));
  }

  // Distributive Omit so each variant of the union keeps its narrowed shape.
  private request<T>(
    msg: InMsg extends infer U ? (U extends { id: number } ? Omit<U, "id"> : never) : never,
    onProgress?: (p: LoadProgress) => void,
  ): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        onProgress,
      });
      this.worker.postMessage({ ...msg, id });
    });
  }

  static async load(onProgress: (p: LoadProgress) => void): Promise<NeedleModel> {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const m = new NeedleModel(worker);
    const meta = await m.request<{ contrastiveDim: number }>({ type: "load" }, onProgress);
    m.contrastiveAvailable = meta.contrastiveDim > 0;
    if (m.contrastiveAvailable) {
      console.info(`[needle-playground] contrastive head present, dim=${meta.contrastiveDim}`);
    } else {
      console.warn(
        "[needle-playground] loaded weights have no contrastive head — " +
          "retrieve_tools will return empty; falling back to JS term-overlap ranker.",
      );
    }
    return m;
  }

  hasContrastiveHead(): boolean {
    return this.contrastiveAvailable;
  }

  infer(query: string, toolsJson: string): Promise<string> {
    return this.request<string>({ type: "infer", query, toolsJson });
  }

  retrieve(
    query: string,
    descriptions: string[],
    topK: number,
  ): Promise<Array<{ index: number; score: number }>> {
    return this.request<Array<{ index: number; score: number }>>({
      type: "retrieve",
      query,
      descriptions,
      topK,
    });
  }
}
