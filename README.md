# Parametric Modal Reverberation — Companion Code

**Anonymous submission to DAFx26**

## Quick start

Place `featureTable.mat` in `data/`, then:

```matlab
demo_figures          % Reproduce all paper figures and tables
demo_paper_examples   % Generate the 3 paper IRs, write WAVs, play back
demo_custom_reverb    % Design your own reverb — edit 6 numbers and run
```

## Demo video

`DAFxVerbDemo.mp4` is a screen recording of the reverberator running
in real time inside a Logic Pro's convolver.  

## Folder structure

```
├── README.md
├── DAFxVerbDemo.mp4          ← DAW demo (watch)
├── demo_figures.m            ← reproduces Figures 1–4 and Tables 2, 4
├── demo_paper_examples.m     ← generates the 3 paper WAVs
├── demo_custom_reverb.m      ← edit parameters, run, listen
├── core/
│   ├── build_parametric_model.m      PCA + IRLS regressions + FIR template
│   ├── parametric_reverb_generate.m  6 controls → modal parameters
│   ├── IR_Synt.m                     exact exponential integrator
│   ├── robustRegress.m               bisquare IRLS with MAD pre-screen
│   └── analyze_correlations.m        correlation analysis + figure generation
├── data/
│   ├── featureTable.mat              ← place here (50 features × 1151 IRs)
│   └── paramModel.mat                  (auto-generated on first run)
└── generated_IRs/                      (auto-created, WAV output)
```

## The six controls

| Control | Range | What it does |
|---------|-------|-------------|
| `T60` | 0.05+ s | Mid-frequency reverberation time |
| `roomSize` | 0–1 | Small → large (maps to centre time) |
| `warmth` | 0.5–2.0 | Bass ratio; >1 = warmer |
| `brightness` | 0.3–2.0 | Treble ratio; >1 = brighter |
| `diffusion` | 0–1 | 0 = metallic grid, 1 = natural spacing |
| `earlyLate` | 0–1 | 0 = diffuse tail, 1 = prominent early reflections |

## Requirements

MATLAB R2020b or later. No toolboxes required.

## Auditioning with your own audio

```matlab
[x, fsx] = audioread('your_file.wav');
ir = audioread('generated_IRs/Medium_Room.wav');
y = conv(x, ir);
soundsc(y, fsx);
```
