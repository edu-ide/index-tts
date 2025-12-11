"""
Speaker Classifier for IndexTTS2 Stage 2 emotion disentanglement.

Based on IndexTTS2 (arXiv:2506.21619v2) Stage 2:
- Connected after GRL (Gradient Reversal Layer)
- Trained to classify speaker identity from emotion vectors
- GRL forces emotion encoder to remove speaker-specific information
- Enables speaker-emotion disentanglement

Architecture:
    Emotion Vector (1280-dim)
        ↓
    GRL (gradient reversal)
        ↓
    Speaker Classifier (1280 → hidden → num_speakers)
        ↓
    Speaker ID prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerClassifier(nn.Module):
    """
    Speaker classifier for emotion-speaker disentanglement.

    This classifier is trained to predict speaker identity from emotion vectors.
    When connected after a GRL, it forces the emotion encoder to produce
    speaker-invariant features.

    Architecture:
        Input (emotion_dim=1280)
        → Linear(1280 → hidden_dim)
        → ReLU
        → Dropout
        → Linear(hidden_dim → num_speakers)
        → Logits

    Loss: CrossEntropyLoss(speaker_logits, speaker_labels)
    """

    def __init__(
        self,
        emotion_dim=1280,
        hidden_dim=512,
        num_speakers=500,  # Adjust based on dataset
        dropout=0.3
    ):
        """
        Initialize speaker classifier.

        Args:
            emotion_dim: Dimension of emotion vectors (default: 1280)
            hidden_dim: Hidden layer dimension (default: 512)
            num_speakers: Number of unique speakers in dataset (default: 500)
            dropout: Dropout probability (default: 0.3)
        """
        super(SpeakerClassifier, self).__init__()

        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.num_speakers = num_speakers

        # Two-layer MLP classifier
        self.fc1 = nn.Linear(emotion_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_speakers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, emo_vec):
        """
        Classify speaker from emotion vector.

        Args:
            emo_vec: Emotion vectors (batch, emotion_dim)

        Returns:
            Speaker logits (batch, num_speakers)
        """
        # Hidden layer
        h = F.relu(self.fc1(emo_vec))
        h = self.dropout(h)

        # Output logits
        logits = self.fc2(h)

        return logits


class SpeakerClassificationLoss(nn.Module):
    """
    Speaker classification loss for Stage 2 training.

    Combined with GRL, this loss enables speaker-emotion disentanglement:
        total_loss = tts_loss + speaker_loss
        (GRL automatically handles the minus sign in gradients)
    """

    def __init__(self, label_smoothing=0.1):
        """
        Initialize loss.

        Args:
            label_smoothing: Label smoothing for CrossEntropyLoss
                            Helps prevent overconfident predictions
        """
        super(SpeakerClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, speaker_logits, speaker_labels):
        """
        Compute speaker classification loss.

        Args:
            speaker_logits: Predicted speaker logits (batch, num_speakers)
            speaker_labels: Ground truth speaker IDs (batch,)

        Returns:
            CrossEntropy loss scalar
        """
        return self.criterion(speaker_logits, speaker_labels)


def build_speaker_id_mapping(manifest_path, max_speakers=None):
    """
    Build speaker ID to integer mapping from manifest file.

    Args:
        manifest_path: Path to train_manifest.jsonl
        max_speakers: Maximum number of speakers to include (for debugging)

    Returns:
        dict: {speaker_name: speaker_id}

    Example:
        >>> mapping = build_speaker_id_mapping('/path/to/train_manifest.jsonl')
        >>> print(f"Total speakers: {len(mapping)}")
        >>> speaker_id = mapping['KO_00OZm1hPrjE_SPEAKER_00']
    """
    import json

    speakers = set()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            speakers.add(data['speaker'])

            if max_speakers and len(speakers) >= max_speakers:
                break

    # Sort for deterministic mapping
    speakers = sorted(list(speakers))

    # Create mapping
    speaker_to_id = {spk: idx for idx, spk in enumerate(speakers)}

    return speaker_to_id


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Speaker Classifier Test")
    print("=" * 60)

    # Create classifier
    num_speakers = 500
    classifier = SpeakerClassifier(
        emotion_dim=1280,
        hidden_dim=512,
        num_speakers=num_speakers,
        dropout=0.3
    )

    print(f"Emotion dim: 1280")
    print(f"Hidden dim: 512")
    print(f"Num speakers: {num_speakers}")
    print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Test forward pass
    batch_size = 16
    emo_vec = torch.randn(batch_size, 1280)
    speaker_labels = torch.randint(0, num_speakers, (batch_size,))

    logits = classifier(emo_vec)
    print(f"\nInput shape: {emo_vec.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {num_speakers})")

    # Test loss
    loss_fn = SpeakerClassificationLoss(label_smoothing=0.1)
    loss = loss_fn(logits, speaker_labels)
    print(f"\nSpeaker classification loss: {loss.item():.4f}")

    # Test accuracy computation
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == speaker_labels).float().mean()
    print(f"Random accuracy: {accuracy.item():.4f} (expected ~{1/num_speakers:.4f})")

    # Test with GRL integration
    print(f"\n{'='*60}")
    print("GRL + Speaker Classifier Integration Test")
    print(f"{'='*60}")

    from gradient_reversal import GradientReversalLayer

    grl = GradientReversalLayer(lambda_=1.0)

    # Simulate emotion encoder output
    emo_vec = torch.randn(batch_size, 1280, requires_grad=True)

    # Apply GRL
    emo_vec_reversed = grl(emo_vec)

    # Speaker classification
    speaker_logits = classifier(emo_vec_reversed)
    speaker_loss = loss_fn(speaker_logits, speaker_labels)

    print(f"Emotion vector shape: {emo_vec.shape}")
    print(f"After GRL shape: {emo_vec_reversed.shape}")
    print(f"Speaker logits shape: {speaker_logits.shape}")
    print(f"Speaker loss: {speaker_loss.item():.4f}")

    # Test backward pass
    speaker_loss.backward()

    print(f"\nGradient flow test:")
    print(f"Emotion vector gradient (should be reversed):")
    print(f"  Mean: {emo_vec.grad.mean().item():.4f}")
    print(f"  Std: {emo_vec.grad.std().item():.4f}")
    print(f"  Min: {emo_vec.grad.min().item():.4f}")
    print(f"  Max: {emo_vec.grad.max().item():.4f}")

    print(f"\n✅ All tests passed!")
