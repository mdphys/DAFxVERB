%% DEMO_FIGURES  Reproduce all figures and tables from the paper.
%
%   Requires: data/featureTable.mat
%
%   Generates (in current directory):
%     corr_matrix.png          — Full 50-feature correlation matrix
%     param_histograms.png     — Figure 1: corpus distributions
%     param_orthogonality.png  — Figure 2: Pearson r, PCA scree, PC scores
%     fit_scatter.png          — Figure 3: predicted-vs-actual (2x5)
%     fit_quality.png          — Full 4x5 diagnostics with residuals
%     synthesis_comparison.png — Figure 4: 3 synthesis examples
%
%   Prints to console:
%     Full-matrix PCA thresholds (Table 2)
%     Regression quality (Table 4)

clear; close all; clc;
addpath('core');
fs = 44100;

%% 1) Load feature table
featureFile = fullfile('data', 'featureTable.mat');
fprintf('=== Loading %s ===\n', featureFile);
assert(exist(featureFile, 'file') > 0, ...
    'featureTable.mat not found in data/.');
tmp = load(featureFile, 'featureTable');
featureTable = tmp.featureTable;
fprintf('  %d IRs, %d features\n', height(featureTable), width(featureTable));

%% 2) Correlation analysis + all diagnostic figures
fprintf('\n=== Correlation analysis and figure generation ===\n');
modelData = analyze_correlations(featureTable, true);

%% 3) Build parametric model
fprintf('\n=== Building parametric model ===\n');
modelFile = fullfile('data', 'paramModel.mat');
paramModel = build_parametric_model(featureFile, modelFile);

%% 4) Synthesis examples (Figure 4)
fprintf('\n=== Synthesis examples ===\n');

%   Name              T60   size  BR    TR    diff  earlyLate
examples = {
    'Small_Dry',      0.4,  0.1,  1.0,  1.0,  0.5,  0.80
    'Medium_Room',    1.2,  0.5,  1.2,  1.0,  0.75, 0.45
    'Large_Hall',     3.0,  0.85, 1.25, 0.8,  0.9,  0.20
};

Nex = size(examples, 1);
figure('Position', [50 50 1200 950]);

for ex = 1:Nex
    up = struct('T60', examples{ex,2}, 'roomSize', examples{ex,3}, ...
                'warmth', examples{ex,4}, 'brightness', examples{ex,5}, ...
                'diffusion', examples{ex,6}, 'earlyLate', examples{ex,7}, 'fs', fs);

    Tsim = max(2, 2 * up.T60);
    [a, b, c, f, hFIR] = parametric_reverb_generate(paramModel, up);
    ir = IR_Synt(a, b, f, c, fs, Tsim, 1, 0);
    ir = conv(ir, hFIR);
    ir = ir(1:round(fs * Tsim));
    ir = ir / (max(abs(ir)) + eps);
    t  = (0:numel(ir)-1) / fs;

    subplot(3, Nex, ex);
    plot(t, ir, 'Color', [0.2 0.4 0.8], 'LineWidth', 0.5);
    xlim([0 Tsim]);
    title(sprintf('%s\nT60=%.1fs  earlyLate=%.2f', ...
        strrep(examples{ex,1}, '_', '\_'), up.T60, up.earlyLate), ...
        'FontSize', 11, 'FontWeight', 'bold');
    if ex == 1, ylabel('Amplitude', 'FontSize', 11); end
    xlabel('Time (s)', 'FontSize', 11); grid on;
    set(gca, 'FontSize', 10);

    subplot(3, Nex, Nex + ex);
    spectrogram(ir, 1024, 768, 1024, fs, 'yaxis');
    ylim([0 12]);
    title(sprintf('BR=%.2f  TR=%.2f  diff=%.2f', ...
        up.warmth, up.brightness, up.diffusion), 'FontSize', 10);
    if ex == 1, ylabel('Frequency (kHz)', 'FontSize', 11); end
    xlabel('Time (s)', 'FontSize', 11);
    set(gca, 'FontSize', 10);

    subplot(3, Nex, 2*Nex + ex); hold on;
    yyaxis left;
    scatter(f, 6.908 ./ c, 3, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.25);
    ylabel('Modal T_{60}  (s)', 'FontSize', 11);
    set(gca, 'YColor', [0.2 0.4 0.8]);
    yyaxis right;
    scatter(f, sqrt(a.^2 + b.^2), 3, [0.8 0.2 0.2], 'filled', 'MarkerFaceAlpha', 0.25);
    ylabel('|r_k|', 'FontSize', 11);
    set(gca, 'YColor', [0.8 0.2 0.2]);
    set(gca, 'XScale', 'log', 'FontSize', 10);
    xlabel('Frequency (Hz)', 'FontSize', 11);
    xlim([40 12000]); xticks([100 1000 10000]); xticklabels({'100','1k','10k'});
    hold off; grid on;
end

sgtitle('Parametric Modal Reverb — Synthesis Examples (3 presets)', 'FontSize', 13);
exportgraphics(gcf, 'synthesis_comparison.png', 'Resolution', 200);
fprintf('  Saved: synthesis_comparison.png\n');

%% 5) Fit quality summary (Table 4)
fprintf('\n=== FIT QUALITY SUMMARY (Table 4) ===\n');
fq = paramModel.fitQuality;
bandNames = {'subBass', 'bass', 'mid', 'presence', 'brilliance'};
fprintf('%-28s  %8s  %8s  %10s\n', 'Target', 'R2_OLS', 'R2_rob', 'n_out');
fprintf('%s\n', repmat('-', 1, 60));
for k = 1:5
    fprintf('  log(sigma) %-12s  %8.3f  %8.3f  %10d\n', bandNames{k}, ...
        fq.R2_sigma_ols(k), fq.R2_sigma_robust(k), fq.nOut_sigma(k));
end
fprintf('  Equipartition            k = %.4f  (std = %.4f)\n', ...
    fq.amp_equip_scale, fq.amp_equip_std);
for k = 1:5
    fprintf('  log(density) %-10s  %8.3f  %8.3f  %10d\n', bandNames{k}, ...
        fq.R2_density_ols(k), fq.R2_density_robust(k), fq.nOut_density(k));
end
fprintf('  FIR energy (C80)         R2 = 1.000  (exact identity)\n');

fprintf('\n=== ALL FIGURES GENERATED ===\n');
