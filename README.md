<div align="center">
<h1>GroundControl: Satellite-Conditioned Multi-View Diffusion<br>for Reconstruction-Free Novel View Synthesis</h1>

<a href="https://nianticspatial.github.io/groundcontrol/resources/groundcontrol.pdf">
  <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2610.37545">
  <img src="https://img.shields.io/badge/arXiv-2610.37545-b31b1b" alt="arXiv">
</a>
<a href="https://nianticspatial.github.io/groundcontrol/">
  <img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page">
</a>
<a href="https://github.com/nianticspatial/groundcontrol">
  <img src="https://img.shields.io/badge/GitHub-Code-black" alt="GitHub">
</a>

**[Niantic Spatial](https://www.nianticspatial.com/)**; **[Georgia Institute of Technology](https://www.cc.gatech.edu/)**; **[Aalto University](https://www.aalto.fi/)**

[Akshay Krishnan](https://akshay-krishnan.github.io)<sup>1,2</sup>,
[Matias Turkulainen](https://maturk.github.io/)<sup>1,3</sup>,
[Jamie Wynn](https://scholar.google.com/citations?user=ASP-uu4AAAAJ&hl=en)<sup>1</sup>,
[Zawar Qureshi](https://scholar.google.cat/citations?user=P2jZC8YAAAAJ&hl=de)<sup>1</sup>,
[Mohamed Sayed](https://masayed.com)<sup>1</sup>,
[Guillermo Garcia-Hernando](https://guiggh.github.io)<sup>1</sup>,
[Sara Vicente](https://scholar.google.com/citations?user=7wWsNNcAAAAJ&hl=en)<sup>1</sup>,
[James Hays](https://www.cc.gatech.edu/~hays/)<sup>2</sup>,
[Michael Firman](https://michaelfirman.co.uk/)<sup>1</sup>,
[Daniyar Turmukhambetov](https://dantkz.github.io/)<sup>1</sup>

<sup>1</sup>Niantic Spatial &nbsp; <sup>2</sup>Georgia Institute of Technology &nbsp; <sup>3</sup>Aalto University

</div>

GroundControl is a satellite-conditioned multi-view diffusion model for outdoor novel-view synthesis.
Given sparse posed street-level photos and a satellite image, it generates target views that follow the
satellite scene layout—especially when the target has little overlap with the inputs—without reconstructing a 3D scene.

This repository releases inference code for Metropolis. See the
[project page](https://nianticspatial.github.io/groundcontrol/) for paper, videos, and results.
Weights and datasets are not included.

## Setup

```bash
mamba env create -f environment.yml
mamba activate groundcontrol
pip install -e .
```

If the CUDA build in `environment.yml` does not match your machine, install PyTorch separately, then `pip install -r requirements.txt`.

## Configure paths

Copy the example paths file and point each entry at your local data and weights:

```bash
cp src/osm_diffusion/configs/paths.example.yaml \
   src/osm_diffusion/configs/paths.yaml
```

`paths.yaml` is gitignored. Model/run settings stay in
`src/osm_diffusion/configs/infer_metropolis.yaml`
(model `satellite_ray_guided_ca`, dual CFG sat=3 / img=2).

## Run

```bash
python -m osm_diffusion.training.eval \
  --config src/osm_diffusion/configs/infer_metropolis.yaml
```

Optional overrides: `--checkpoint-path`, `--output-path`, `--limit-val-batches`.

```bash
bash scripts/infer_metropolis.sh
bash scripts/infer_metropolis.sh --limit-val-batches 1
```

## Citation

```bibtex
@inproceedings{krishnan2027groundcontrol,
  title     = {{GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis}},
  author    = {Krishnan, Akshay and Turkulainen, Matias and Wynn, Jamie and Qureshi, Zawar and Sayed, Mohamed and Garcia-Hernando, Guillermo and Vicente, Sara and Hays, James and Firman, Michael and Turmukhambetov, Daniyar},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
