%% DEMO_PAPER_EXAMPLES  Generate the three IRs from the paper (Figure 4).
clear; close all; clc;
addpath('core');

fs     = 44100;
outDir = './generated_IRs';
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% Build or load model
modelFile   = fullfile('data', 'paramModel.mat');
featureFile = fullfile('data', 'featureTable.mat');
if exist(modelFile, 'file')
    fprintf('Loading cached model: %s\n', modelFile);
    load(modelFile, 'paramModel');
else
    fprintf('Building model from %s ...\n', featureFile);
    paramModel = build_parametric_model(featureFile, modelFile);
end

%% Paper presets (Figure 4)
%   Name              T60   size   BR     TR    diff  earlyLate
presets = {
    'Small_Dry',     0.40,  0.10,  1.00,  1.00,  0.50,  0.80
    'Medium_Room',   1.20,  0.50,  1.20,  1.00,  0.75,  0.45
    'Large_Hall',    3.00,  0.85,  1.25,  0.80,  0.90,  0.20
};

Np = size(presets, 1);
fprintf('\n=== Generating %d IRs ===\n\n', Np);

for p = 1:Np
    up = struct('T60', presets{p,2}, 'roomSize', presets{p,3}, ...
                'warmth', presets{p,4}, 'brightness', presets{p,5}, ...
                'diffusion', presets{p,6}, 'earlyLate', presets{p,7}, 'fs', fs);

    Tsim = max(2, 2 * up.T60);
    [a, b, c, f, hFIR] = parametric_reverb_generate(paramModel, up);
    ir = IR_Synt(a, b, f, c, fs, Tsim, 1, 0);
    ir = conv(ir, hFIR);
    ir = ir(1:round(fs * Tsim));
    ir = ir / (max(abs(ir)) + eps);

    wavName = [presets{p,1} '.wav'];
    audiowrite(fullfile(outDir, wavName), ir, fs);
    fprintf('  [%d] %s  (T60=%.1fs)\n', p, wavName, up.T60);
end

fprintf('\n%d WAV files written to: %s\n', Np, outDir);

%% Playback
fprintf('\nPlaying back (press Ctrl-C to stop)...\n\n');
for p = 1:Np
    fprintf('  Playing: %s\n', presets{p,1});
    [y, fsr] = audioread(fullfile(outDir, [presets{p,1} '.wav']));
    sound(y, fsr);
    pause(max(1.5, presets{p,2}));
end
fprintf('\nDone.\n');
