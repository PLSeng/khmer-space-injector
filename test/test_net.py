from pathlib import Path
import torch

from src.net import KhmerRNN

torch.manual_seed(0)

# Paths: test/testdata/*.txt
TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "testdata"
ORIG_PATH = DATA_DIR / "313328_orig.txt"
SEG_PATH  = DATA_DIR / "313328_seg.txt"


def _read_first_nonempty_line(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No non-empty lines in {path}")


def _strip_all_whitespace(s: str) -> str:
    # removes spaces, double spaces, tabs, etc.
    return "".join(s.split())


def test_data_pair_alignment():
    """
    Khmer word segmentation invariant:
    segmented text, after removing ALL whitespace, must equal original text.
    """
    orig = _read_first_nonempty_line(ORIG_PATH)
    seg  = _read_first_nonempty_line(SEG_PATH)

    assert _strip_all_whitespace(orig) == _strip_all_whitespace(seg), \
        "orig and seg don't match after removing all whitespace"


def test_forward_no_crf_shape():
    model = KhmerRNN(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        use_crf=False,
    )
    model.eval()

    B, T = 4, 20
    x = torch.randint(1, 100, (B, T))

    with torch.no_grad():
        emissions = model(x)

    assert emissions.shape == (B, T, 2)


def test_backward_no_crf_gradients():
    model = KhmerRNN(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        use_crf=False,
    )
    model.train()

    B, T = 4, 20
    x = torch.randint(1, 100, (B, T))
    y = torch.randint(0, 2, (B, T))

    emissions = model(x)  # (B,T,2)
    loss = torch.nn.CrossEntropyLoss()(emissions.reshape(-1, 2), y.reshape(-1))
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    ), "no valid gradients found"


def test_save_and_load_state_dict(tmp_path):
    model1 = KhmerRNN(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        use_crf=False,
    )
    model1.eval()

    x = torch.randint(1, 100, (2, 10))
    with torch.no_grad():
        out1 = model1(x)

    ckpt = tmp_path / "net.pt"
    torch.save(model1.state_dict(), ckpt)

    model2 = KhmerRNN(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        use_crf=False,
    )
    model2.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model2.eval()

    with torch.no_grad():
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)
