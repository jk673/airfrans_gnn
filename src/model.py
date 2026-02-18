# 2b) Enhanced Global Context & Attention Mechanism
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""
    def __init__(self, d_model, max_len=50000):  # Increased max_len for large graphs
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.max_len = max_len

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # For very large sequences, use a learnable encoding or skip
            return x
        return x + self.pe[:, :seq_len]

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional temperature scaling"""
    def __init__(self, d_model, num_heads, dropout=0.1, temperature_scaling=True, bias=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature_scaling = temperature_scaling

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k) if temperature_scaling else 1.0

    def forward(self, query, key, value, mask=None):
        batch_size, q_seq_len, _ = query.size()
        _, k_seq_len, _ = key.size()
        _, v_seq_len, _ = value.size()

        # Linear projections
        q = self.w_q(query).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)

        return self.w_o(context), attn_weights

class CrossAttention(nn.Module):
    """Cross-attention between local node features and global tokens"""
    def __init__(self, d_model, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout, bias=bias)

    def forward(self, nodes, globals, mask=None):
        # nodes: [batch_size, num_nodes, d_model]
        # globals: [batch_size, num_global_tokens, d_model]

        # Cross-attention: nodes attend to global tokens (no mask for now)
        node_context, node_attn = self.attention(nodes, globals, globals, None)

        # Cross-attention: global tokens attend to nodes (no mask for now)
        global_context, global_attn = self.attention(globals, nodes, nodes, None)

        return node_context, global_context, (node_attn, global_attn)

class GatedGCNLayer(nn.Module):
    """Gated Graph Convolution Layer for local updates"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear transformations for gating
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [num_nodes, hidden_dim] - node features
        edge_index: [2, num_edges] - edge indices
        edge_attr: [num_edges, hidden_dim] - edge features (optional)
        """
        row, col = edge_index

        # Aggregate messages from neighbors
        messages = x[row]  # Messages from source nodes
        if edge_attr is not None:
            messages = messages + edge_attr

        # Aggregate by target nodes
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, messages)

        # Compute gates and updates
        combined = torch.cat([x, aggregated], dim=1)
        gate = torch.sigmoid(self.gate_linear(combined))
        update = torch.tanh(self.update_linear(combined))

        # Apply gated update with residual connection
        new_x = gate * update + (1 - gate) * x
        new_x = self.norm(new_x)
        new_x = self.dropout(new_x)

        return new_x

class TransformerLayer(nn.Module):
    """Transformer layer for global context modeling"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, ff_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim or hidden_dim * 4

        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, hidden_dim]
        mask: optional attention mask
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff_network(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights

class EnhancedGlobalContextProcessor(nn.Module):
    """Enhanced global context processor supporting two modes:

    - 'global_tokens' (default): Uses learnable global tokens, cross-attention,
      and positional encoding for global context.
    - 'gated_gcn_transformer': Uses GatedGCN layers for local graph updates
      followed by Transformer layers for global context modeling.
    """

    def __init__(self, hidden_dim=128, num_global_tokens=4, num_heads=8,
                 num_layers=2, dropout=0.1, use_cross_attention=True,
                 global_pooling_type='attention', use_positional_encoding=True,
                 use_residual=True, norm_type='layer', temperature_scaling=True,
                 pos_max_len=50000, context_mode='global_tokens',
                 num_gcn_layers=None, num_transformer_layers=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.global_pooling_type = global_pooling_type
        self.context_mode = context_mode

        if context_mode == 'global_tokens':
            # --- Original global tokens mode ---
            self.num_global_tokens = num_global_tokens
            self.use_cross_attention = use_cross_attention
            self.use_positional_encoding = use_positional_encoding

            # Learnable global tokens
            self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, hidden_dim) * 0.02)
            nn.init.xavier_uniform_(self.global_tokens, gain=1.0)

            # Positional encoding
            if use_positional_encoding:
                self.pos_encoding = PositionalEncoding(hidden_dim, max_len=pos_max_len)

            # Self-attention layers for global tokens
            self.global_self_attention = nn.ModuleList([
                MultiHeadSelfAttention(hidden_dim, num_heads, dropout, temperature_scaling)
                for _ in range(num_layers)
            ])

            # Cross-attention between nodes and global tokens
            if use_cross_attention:
                self.cross_attention = nn.ModuleList([
                    CrossAttention(hidden_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ])

            # Normalization layers
            if norm_type == 'layer':
                self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
                self.global_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            elif norm_type == 'batch':
                self.node_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
                self.global_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
            else:
                self.node_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
                self.global_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

            # Global pooling mechanisms
            if global_pooling_type == 'attention':
                self.global_pooling = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
                self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            elif global_pooling_type == 'set2set':
                self.global_pooling = Set2Set(hidden_dim, processing_steps=3)

            # Output projections
            self.node_output_proj = nn.Linear(hidden_dim, hidden_dim)
            self.global_output_proj = nn.Linear(hidden_dim, hidden_dim)

        elif context_mode == 'gated_gcn_transformer':
            # --- GatedGCN + Transformer mode ---
            _num_gcn = num_gcn_layers if num_gcn_layers is not None else num_layers
            _num_tf = num_transformer_layers if num_transformer_layers is not None else num_layers
            self.num_gcn_layers = _num_gcn
            self.num_transformer_layers = _num_tf

            # GatedGCN layers for local updates
            self.gcn_layers = nn.ModuleList([
                GatedGCNLayer(hidden_dim, dropout)
                for _ in range(_num_gcn)
            ])

            # Transformer layers for global context
            self.transformer_layers = nn.ModuleList([
                TransformerLayer(hidden_dim, num_heads, dropout)
                for _ in range(_num_tf)
            ])

            # Global pooling mechanism
            if global_pooling_type == 'attention':
                self.global_pooling = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
                self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            elif global_pooling_type == 'set2set':
                self.global_pooling = Set2Set(hidden_dim, processing_steps=3)

            # Output projection (concatenates local + global, so hidden_dim*2 -> hidden_dim)
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(f"Unknown context_mode: {context_mode!r}. "
                             f"Expected 'global_tokens' or 'gated_gcn_transformer'.")

        # Dropout (shared)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, batch=None, edge_index=None, edge_attr=None):
        """
        node_features: [num_nodes, hidden_dim]
        batch: batch indices for nodes (for batched processing)
        edge_index: [2, num_edges] - edge connectivity (required for gated_gcn_transformer mode)
        edge_attr: [num_edges, hidden_dim] - edge features (optional, used by gated_gcn_transformer mode)
        """
        if self.context_mode == 'global_tokens':
            return self._forward_global_tokens(node_features, batch)
        else:
            return self._forward_gated_gcn_transformer(node_features, edge_index, edge_attr, batch)

    def _forward_global_tokens(self, node_features, batch=None):
        """Original global tokens + cross-attention forward pass."""
        batch_size = 1 if batch is None else batch.max().item() + 1
        device = node_features.device

        # Expand global tokens for batch
        global_tokens = self.global_tokens.expand(batch_size, -1, -1)  # [batch_size, num_global_tokens, hidden_dim]

        # Prepare node features for batched processing
        if batch is not None:
            # Convert to batched format
            max_nodes = torch.bincount(batch).max().item()
            batched_nodes = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
            node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

            for i in range(batch_size):
                mask = (batch == i)
                num_nodes_i = mask.sum().item()
                if num_nodes_i > 0:
                    batched_nodes[i, :num_nodes_i] = node_features[mask]
                    node_mask[i, :num_nodes_i] = True
        else:
            batched_nodes = node_features.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            node_mask = torch.ones(1, node_features.size(0), dtype=torch.bool, device=device)

        # Add positional encoding
        if self.use_positional_encoding:
            batched_nodes = self.pos_encoding(batched_nodes)
            global_tokens = self.pos_encoding(global_tokens)

        # Multi-layer attention processing
        all_attention_weights = []

        for i, (global_self_attn, node_norm, global_norm) in enumerate(
            zip(self.global_self_attention, self.node_norms, self.global_norms)):

            # Global self-attention
            global_residual = global_tokens
            global_tokens_attn, global_attn_weights = global_self_attn(global_tokens, global_tokens, global_tokens)

            if self.use_residual:
                global_tokens = global_residual + self.dropout(global_tokens_attn)
            else:
                global_tokens = global_tokens_attn

            global_tokens = global_norm(global_tokens)

            # Cross-attention between nodes and global tokens
            if self.use_cross_attention and i < len(self.cross_attention):
                node_residual = batched_nodes

                node_context, global_context, cross_attn_weights = self.cross_attention[i](
                    batched_nodes, global_tokens, None)

                if self.use_residual:
                    batched_nodes = node_residual + self.dropout(node_context)
                    global_tokens = global_tokens + self.dropout(global_context)
                else:
                    batched_nodes = node_context
                    global_tokens = global_context

                batched_nodes = node_norm(batched_nodes)
                global_tokens = global_norm(global_tokens)

                all_attention_weights.append({
                    'global_self': global_attn_weights,
                    'cross': cross_attn_weights
                })

        # Global pooling to create per-node global context
        if self.global_pooling_type == 'attention':
            # Attention-based pooling
            pooled_global, pool_attn = self.global_pooling(
                self.pool_query.expand(batch_size, -1, -1),
                global_tokens, global_tokens
            )
            # Broadcast to all nodes
            global_context_per_node = pooled_global.expand(-1, batched_nodes.size(1), -1)

        elif self.global_pooling_type == 'mean':
            # Mean pooling
            global_context_per_node = global_tokens.mean(dim=1, keepdim=True).expand(-1, batched_nodes.size(1), -1)

        elif self.global_pooling_type == 'max':
            # Max pooling
            global_context_per_node = global_tokens.max(dim=1, keepdim=True)[0].expand(-1, batched_nodes.size(1), -1)

        else:
            # Simple broadcast
            global_context_per_node = global_tokens.mean(dim=1, keepdim=True).expand(-1, batched_nodes.size(1), -1)

        # Output projections
        enhanced_nodes = self.node_output_proj(batched_nodes + global_context_per_node)
        enhanced_globals = self.global_output_proj(global_tokens)

        # Convert back to flat format if needed
        if batch is not None:
            output_nodes = torch.zeros_like(node_features)
            for i in range(batch_size):
                mask = (batch == i)
                num_nodes_i = mask.sum().item()
                if num_nodes_i > 0:
                    # Ensure dtype compatibility for AMP
                    output_nodes[mask] = enhanced_nodes[i, :num_nodes_i].to(output_nodes.dtype)
        else:
            output_nodes = enhanced_nodes.squeeze(0)

        return output_nodes, enhanced_globals, all_attention_weights

    def _forward_gated_gcn_transformer(self, node_features, edge_index, edge_attr=None, batch=None):
        """GatedGCN local updates + Transformer global context forward pass."""
        device = node_features.device
        x = node_features

        # Phase 1: Local updates with GatedGCN layers
        for gcn_layer in self.gcn_layers:
            if self.use_residual:
                x = x + gcn_layer(x, edge_index, edge_attr)
            else:
                x = gcn_layer(x, edge_index, edge_attr)

        # Phase 2: Global context with Transformer
        if batch is not None:
            # Convert to batched format for transformer
            batch_size = batch.max().item() + 1
            max_nodes = torch.bincount(batch).max().item()
            batched_nodes = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
            node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

            for i in range(batch_size):
                mask = (batch == i)
                num_nodes_i = mask.sum().item()
                if num_nodes_i > 0:
                    batched_nodes[i, :num_nodes_i] = x[mask]
                    node_mask[i, :num_nodes_i] = True
        else:
            batched_nodes = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            node_mask = torch.ones(1, x.size(0), dtype=torch.bool, device=device)
            batch_size = 1

        # Apply transformer layers for global context
        global_features = batched_nodes
        all_attention_weights = []

        for transformer_layer in self.transformer_layers:
            # Create attention mask from node_mask
            attention_mask = None
            if node_mask is not None:
                # Convert to attention mask format: [batch_size, seq_len, seq_len]
                attention_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension

            global_features, attn_weights = transformer_layer(global_features, attention_mask)
            all_attention_weights.append(attn_weights)

        # Phase 3: Global pooling to create global context
        if self.global_pooling_type == 'attention':
            # Attention-based pooling
            pooled_global, pool_attn = self.global_pooling(
                self.pool_query.expand(batch_size, -1, -1),
                global_features, global_features,
                key_padding_mask=~node_mask if node_mask is not None else None
            )
            # Broadcast to all nodes
            global_context = pooled_global.expand(-1, global_features.size(1), -1)

        elif self.global_pooling_type == 'mean':
            # Mean pooling with mask
            if node_mask is not None:
                global_sum = (global_features * node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
                global_count = node_mask.sum(dim=1, keepdim=True).float().unsqueeze(-1)
                global_context = (global_sum / global_count.clamp(min=1)).expand(-1, global_features.size(1), -1)
            else:
                global_context = global_features.mean(dim=1, keepdim=True).expand(-1, global_features.size(1), -1)
        else:
            # Default to mean pooling
            global_context = global_features.mean(dim=1, keepdim=True).expand(-1, global_features.size(1), -1)

        # Phase 4: Combine local and global features
        combined_features = torch.cat([global_features, global_context], dim=-1)
        enhanced_nodes = self.output_proj(combined_features)

        # Convert back to flat format
        if batch is not None:
            output_nodes = torch.zeros_like(node_features)
            for i in range(batch_size):
                mask = (batch == i)
                num_nodes_i = mask.sum().item()
                if num_nodes_i > 0:
                    output_nodes[mask] = enhanced_nodes[i, :num_nodes_i].to(output_nodes.dtype)
        else:
            output_nodes = enhanced_nodes.squeeze(0)

        return output_nodes, global_features, all_attention_weights

class Set2Set(nn.Module):
    """Set2Set pooling mechanism"""
    def __init__(self, input_dim, processing_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTM(input_dim * 2, input_dim, batch_first=True)

    def forward(self, x):
        batch_size, set_size, dim = x.size()

        h = torch.zeros(1, batch_size, self.input_dim, device=x.device)
        c = torch.zeros(1, batch_size, self.input_dim, device=x.device)

        q_star = torch.zeros(batch_size, 1, self.input_dim, device=x.device)

        for _ in range(self.processing_steps):
            # Attention
            attention = torch.bmm(x, q_star.transpose(1, 2))  # [batch, set_size, 1]
            attention = F.softmax(attention, dim=1)

            # Weighted sum
            r = torch.bmm(attention.transpose(1, 2), x)  # [batch, 1, dim]

            # Update with LSTM
            q_input = torch.cat([q_star, r], dim=2)  # [batch, 1, 2*dim]
            q_star, (h, c) = self.lstm(q_input, (h, c))

        return q_star.squeeze(1)  # [batch, dim]


# Create enhanced model with global context configuration
class EnhancedCFDModelWithGlobalContext(nn.Module):
    """CFD model with enhanced global context and attention mechanisms"""

    def __init__(self, node_feat_dim=7, edge_feat_dim=5, hidden_dim=128,
                 output_dim=4, num_mp_layers=14, dropout_p=0.1, config=None):
        super().__init__()

        # Avoid relying on a module-global 'scfg'. Provide safe defaults if config is None.
        if config is None:
            class _DefaultCfg:
                # Global Context & Attention Configuration (safe defaults)
                use_global_tokens = False
                num_global_tokens = 2
                attention_heads = 4
                attention_layers = 1
                attention_dropout = 0.0
                use_cross_attention = False
                global_pooling_type = 'attention'
                positional_encoding = False
                use_residual_attention = True
                attention_normalization = 'layer'
                temperature_scaling = True
                pos_encoding_max_len = 50000
                # GatedGCN + Transformer parameters
                context_mode = 'global_tokens'
                num_gcn_layers = None
                num_transformer_layers = None
            self.config = _DefaultCfg()
        else:
            self.config = config
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim

        # Store context_mode for forward pass routing
        self._context_mode = getattr(self.config, 'context_mode', 'global_tokens')

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

        # Enhanced global context processor
        if self.config.use_global_tokens:
            self.global_context_processor = EnhancedGlobalContextProcessor(
                hidden_dim=hidden_dim,
                num_global_tokens=getattr(self.config, 'num_global_tokens', 4),
                num_heads=self.config.attention_heads,
                num_layers=self.config.attention_layers,
                dropout=self.config.attention_dropout,
                use_cross_attention=getattr(self.config, 'use_cross_attention', True),
                global_pooling_type=self.config.global_pooling_type,
                use_positional_encoding=getattr(self.config, 'positional_encoding', True),
                use_residual=self.config.use_residual_attention,
                norm_type=getattr(self.config, 'attention_normalization', 'layer'),
                temperature_scaling=getattr(self.config, 'temperature_scaling', True),
                pos_max_len=getattr(self.config, 'pos_encoding_max_len', 50000),
                context_mode=self._context_mode,
                num_gcn_layers=getattr(self.config, 'num_gcn_layers', None),
                num_transformer_layers=getattr(self.config, 'num_transformer_layers', None),
            )

        # Message passing layers (simplified version for demonstration)
        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(nn.ModuleDict({
                'edge_update': nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                ),
                'node_update': nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            }))

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )

    def get_node_embeddings(self, data):
        """Return node features as embeddings"""
        return data.x

    def forward(self, data):
        device = data.x.device

        # Encode node and edge features
        x = self.node_encoder(data.x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_encoder(data.edge_attr)  # [num_edges, hidden_dim]

        # Apply enhanced global context processing
        if self.config.use_global_tokens:
            batch = getattr(data, 'batch', None)
            if self._context_mode == 'gated_gcn_transformer':
                x, global_tokens, attention_weights = self.global_context_processor(
                    x, batch, edge_index=data.edge_index, edge_attr=edge_attr)
            else:
                x, global_tokens, attention_weights = self.global_context_processor(x, batch)

            # Store attention weights for analysis
            self.last_attention_weights = attention_weights

        # Message passing
        for layer in self.mp_layers:
            x_residual = x
            edge_residual = edge_attr

            # Edge update
            row, col = data.edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_update = layer['edge_update'](edge_input)
            edge_attr = edge_residual + edge_update

            # Node aggregation and update
            x_agg = torch.zeros_like(x)
            # Ensure dtype compatibility for AMP
            edge_attr_compat = edge_attr.to(x_agg.dtype)
            x_agg.index_add_(0, col, edge_attr_compat)

            # Count neighbors for normalization
            ones = torch.ones(data.edge_index.size(1), 1, device=device, dtype=x.dtype)
            count = torch.zeros(x.size(0), 1, device=device, dtype=x.dtype)
            count.index_add_(0, col, ones)
            x_agg = x_agg / count.clamp(min=1)

            node_input = torch.cat([x_residual, x_agg], dim=1)
            node_update = layer['node_update'](node_input)
            x = x_residual + node_update

        # Decode to output
        output = self.decoder(x)
        return output
