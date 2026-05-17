/// FFN activation function (matches Python `activation` config field).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum FfnActivation {
    #[default]
    DRelu,
    SwiGLU,
    GeGLU,
}

impl FfnActivation {
    pub fn parse(s: &str) -> Self {
        match s {
            "swiglu" => Self::SwiGLU,
            "geglu" => Self::GeGLU,
            _ => Self::DRelu, // "drelu" and unknown → DRelu
        }
    }
}

/// Mirror of Python TransformerConfig — all hyperparameters for Needle SAN.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub num_dec_layers: usize,
    pub vocab_size: usize,
    pub max_enc_len: usize,
    pub max_dec_len: usize,
    pub ffn_dim: usize,
    pub no_feedforward: bool,
    /// FFN activation function (drelu/swiglu/geglu).
    pub activation: FfnActivation,
    /// RoPE base frequency (Python: rope_theta, default 10000.0).
    pub rope_theta: f32,
    /// Special token IDs
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub tool_call_id: u32,
    pub tools_id: u32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 12,
            num_dec_layers: 8,
            vocab_size: 8192,
            max_enc_len: 1024,
            max_dec_len: 512,
            ffn_dim: 2048,
            no_feedforward: true,
            activation: FfnActivation::DRelu,
            rope_theta: 10000.0,
            pad_id: 0,
            eos_id: 1,
            bos_id: 2,
            unk_id: 3,
            tool_call_id: 4,
            tools_id: 5,
        }
    }
}

impl TransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Number of times each KV head is repeated to cover all Q heads.
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Validate structural invariants. Returns an error string if violated.
    /// Called by the inference engine before constructing a model.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_heads == 0 {
            return Err("num_heads must be > 0");
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0");
        }
        if self.d_model == 0 {
            return Err("d_model must be > 0");
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err("d_model must be divisible by num_heads");
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err("num_heads must be divisible by num_kv_heads");
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0");
        }
        Ok(())
    }
}
