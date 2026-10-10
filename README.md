# HeartSSM
Learning a dynamic four-chamber shape model of the human heart for 95,695 UK Biobank participants

Neural HeartFFDNet adds three-view 2D mask/SDF sequences, shape/motion latents,
learned 3D features, physical multiscale FFD and four-chamber 4D supervision.
The original volume-based pipeline remains available unchanged.

```bash
python train_neural_ffd.py --config configs/neural_ffd.yaml
python -m unittest discover -s tests -v
python train_neural_ffd.py --evaluate outputs/neural_ffd/last.pth --exact
```

See [implementation and validation](docs/neural_ffd.rst),
[public-source review](docs/neural_ffd_sources.json), and
[configuration](configs/neural_ffd.yaml). Architecture checks do not establish
trained reconstruction accuracy; formal population training remains to be run.
