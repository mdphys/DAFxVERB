function paramModel = build_parametric_model(source, saveFile)
%BUILD_PARAMETRIC_MODEL  Learn parametric mapping: 6 user controls -> modal coefficients.
%
%   paramModel = build_parametric_model(source)
%   paramModel = build_parametric_model(source, 'paramModel.mat')
%
%   SOURCE can be:
%     (a) Path to featureTable.mat saved by batch_extract_features  [preferred]
%     (b) Path to a folder containing *_RESULTS.mat files  [legacy / re-scan]
%
%   The feature table must contain per-band mean damping (sigma_subBass,
%   ..., sigma_brilliance) and acoustic indices measured on the
%   resynthesised IR.

if nargin < 2, saveFile = ''; end
fprintf('\n========== BUILDING PARAMETRIC MODEL (v9, 6-param, syn-only) ==========\n');

bandEdges   = [40 125; 125 500; 500 2000; 2000 8000; 8000 12000];
Nbands      = size(bandEdges, 1);
bandCenters = sqrt(bandEdges(:,1) .* bandEdges(:,2));
bandNames5  = {'subBass','bass','mid','presence','brilliance'};

%% 1) Load data
if ischar(source) && numel(source)>4 && strcmpi(source(end-3:end),'.mat') && isfile(source)

    % ── Mode (a): load from featureTable.mat ─────────────────────────────
    fprintf('Loading from featureTable: %s\n', source);
    D = load(source, 'featureTable', 'hFIR_all', 'amp_sqrt_sigma_ratios');
    if ~isfield(D,'hFIR_all') || ~isfield(D,'amp_sqrt_sigma_ratios')
        error(['featureTable.mat is missing hFIR_all / amp_sqrt_sigma_ratios.\n' ...
               'Re-run batch_extract_features to regenerate it.']);
    end
    ft     = D.featureTable;
    Nvalid = height(ft);
    fprintf('Loaded %d IRs\n', Nvalid);

    T60v       = ft.T30_mean_mid;
    Tsv        = ft.Ts;
    BRv        = ft.BR;
    TRv        = ft.TR;
    scvAll     = ft.spacing_cv;
    C80v       = ft.C80_mean_mid;
    Nmodes_all = ft.Nmodes;

    sb_all = nan(Nvalid, Nbands);
    db_all = nan(Nvalid, Nbands);
    for b = 1:Nbands
        sb_all(:,b) = ft.(['sigma_'   bandNames5{b}]);
        db_all(:,b) = ft.(['density_' bandNames5{b}]);
    end

    hFIR_all_c              = D.hFIR_all;
    amp_sqrt_sigma_ratios_c = D.amp_sqrt_sigma_ratios;

else

    % ── Mode (b): scan raw _RESULTS.mat files ────────────────────────────
    matFolder = source;
    files  = dir(fullfile(matFolder, '*_RESULTS.mat'));
    Nfiles = numel(files);
    fprintf('Scanning %d _RESULTS.mat files in %s\n', Nfiles, matFolder);

    T60_raw   = nan(Nfiles,1);  Ts_raw  = nan(Nfiles,1);
    BR_raw    = nan(Nfiles,1);  TR_raw  = nan(Nfiles,1);
    scv_raw   = nan(Nfiles,1);  C80_raw = nan(Nfiles,1);
    Nm_raw    = nan(Nfiles,1);
    sb_raw    = nan(Nfiles, Nbands);
    db_raw    = nan(Nfiles, Nbands);
    hFIR_raw  = cell(Nfiles,1);
    ratio_raw = cell(Nfiles,1);

    for iFile = 1:Nfiles
        try, S = load(fullfile(matFolder, files(iFile).name)); catch, continue; end
        fKeep = double(S.fKeep(:));  cKeep = double(S.cKeep(:));
        aKeep = double(S.aKeep(:));  bKeep = double(S.bKeep(:));
        Nm = numel(fKeep);
        if Nm < 10, continue; end
        amp = sqrt(aKeep.^2 + bKeep.^2);

        % Acoustic indices from resynthesised IR
        aiBands = double(S.synAI_final.bandCentersHz(:));
        T30v_s  = double(S.synAI_final.T30(:));
        C80v_s  = double(S.synAI_final.C80(:));
        [~,i5]  = min(abs(log2(aiBands/500)));
        [~,i10] = min(abs(log2(aiBands/1000)));
        T60_raw(iFile) = mean([T30v_s(i5), T30v_s(i10)], 'omitnan');
        Ts_raw(iFile)  = double(S.synAI_final.Ts);
        BR_raw(iFile)  = double(S.synAI_final.BR);
        TR_raw(iFile)  = double(S.synAI_final.TR);
        C80_raw(iFile) = mean([C80v_s(i5), C80v_s(i10)], 'omitnan');
        Nm_raw(iFile)  = Nm;

        for b = 1:Nbands
            mask = fKeep >= bandEdges(b,1) & fKeep < bandEdges(b,2);
            if any(mask)
                sb_raw(iFile,b) = mean(cKeep(mask));
                db_raw(iFile,b) = sum(mask) / (bandEdges(b,2)-bandEdges(b,1));
            end
        end
        if Nm > 2
            sp = diff(sort(fKeep));
            scv_raw(iFile) = std(sp) / (mean(sp)+eps);
        end
        vAS = amp>0 & cKeep>0;
        if sum(vAS)>=3, ratio_raw{iFile} = amp(vAS)./sqrt(cKeep(vAS)); end
        if isfield(S,'hFIR') && numel(S.hFIR)>1, hFIR_raw{iFile} = double(S.hFIR(:)); end
        if mod(iFile,100)==0, fprintf('  Loaded %d/%d\n',iFile,Nfiles); end
    end

    valid  = ~isnan(T60_raw);
    Nvalid = sum(valid);
    fprintf('Valid IRs: %d / %d\n', Nvalid, Nfiles);

    T60v       = T60_raw(valid);   Tsv    = Ts_raw(valid);
    BRv        = BR_raw(valid);    TRv    = TR_raw(valid);
    scvAll     = scv_raw(valid);   C80v   = C80_raw(valid);
    Nmodes_all = Nm_raw(valid);
    sb_all = sb_raw(valid,:);
    db_all = db_raw(valid,:);
    hFIR_all_c              = hFIR_raw(valid);
    amp_sqrt_sigma_ratios_c = ratio_raw(valid);
end

%% 2) PCA orthogonalisation of the 6 model parameters
%   Parameters: [T60, Ts, BR, TR, SpacingCV, C80_mid]
%   All measured on the resynthesised IR.

fprintf('\n--- PCA orthogonalisation of 6 parameters ---\n');

vm_pca = isfinite(T60v) & isfinite(Tsv) & isfinite(BRv) & isfinite(TRv) ...
       & isfinite(scvAll) & isfinite(C80v);

P_raw = [T60v(vm_pca), Tsv(vm_pca), BRv(vm_pca), TRv(vm_pca), scvAll(vm_pca), C80v(vm_pca)];
N_pca = sum(vm_pca);
fprintf('Complete cases for PCA: %d / %d\n', N_pca, Nvalid);

% Robust standardisation
pca_mu    = median(P_raw, 1);
pca_sigma = (prctile(P_raw,75) - prctile(P_raw,25)) / 1.3490;
pca_sigma(pca_sigma < eps) = 1;

P_std = (P_raw - pca_mu) ./ pca_sigma;

[pca_coeff, pca_scores, ~, ~, pca_explained] = pca(P_std);

fprintf('Variance explained per PC: ');
fprintf('%.1f%%  ', pca_explained); fprintf('\n');
fprintf('Cumulative:                ');
fprintf('%.1f%%  ', cumsum(pca_explained)); fprintf('\n');

S_corr = corr(pca_scores);
off = S_corr - eye(6);
fprintf('PC score max |off-diag correlation| = %.2e  (should be ~0)\n', max(abs(off(:))));

%% 3) Robust regressions using PC scores as predictors
fprintf('\n--- Robust regressions (IRLS, c=3.0) with orthogonal PC predictors ---\n');

irlsOpts = struct('maxIter',50,'tol',1e-6,'c',3.0, ...
                  'prescreen',true,'prescreen_k',3.0,'verbose',false);

X_pc = [ones(N_pca,1), pca_scores];  % [N_pca × 7]

% --- 3a) Per-band damping: log(sigma_b) ~ PC scores -------------------------
%   Same structure as density: 5 independent regressions in log space.
%   At synthesis, the 5 predicted band sigmas are interpolated to per-mode
%   frequencies via pchip on log10(f).

sb_pca = sb_all(vm_pca,:);

beta_log_sigma = nan(7, Nbands);
R2_sigma_rob   = nan(1, Nbands);
R2_sigma_ols   = nan(1, Nbands);
nOut_sigma     = nan(1, Nbands);

for b = 1:Nbands
    vmb = ~isnan(sb_pca(:,b)) & sb_pca(:,b) > 0;
    if sum(vmb) > 5
        Xd = X_pc(vmb,:);
        yd = log(sb_pca(vmb, b));
        [bb, ~, r2r, r2o, no] = robustRegress(Xd, yd, irlsOpts);
        beta_log_sigma(:,b) = bb;
        R2_sigma_rob(b) = r2r;  R2_sigma_ols(b) = r2o;  nOut_sigma(b) = no;
    end
end
fprintf('log(Sigma) bands   R2_rob=[%s]\n', num2str(R2_sigma_rob,'%.3f '));

% --- 3b) Amplitude scale — equipartition constant ----------------------------
all_ratios = [];
ratio_cell = amp_sqrt_sigma_ratios_c;
for iFile = 1:numel(ratio_cell)
    if ~isempty(ratio_cell{iFile})
        all_ratios = [all_ratios; ratio_cell{iFile}(:)]; %#ok
    end
end
ratio_med = median(all_ratios,'omitnan');
ratio_mad = mad(all_ratios,1);
ratio_inl = all_ratios(abs(all_ratios - ratio_med) <= 3.0*ratio_mad/0.6745);
amp_equip_scale = median(ratio_inl);
amp_equip_std   = std(ratio_inl);
fprintf('Equip. scale k = %.4f  (std=%.4f, n=%d inliers / %d total modes)\n', ...
    amp_equip_scale, amp_equip_std, numel(ratio_inl), numel(all_ratios));

% --- 3c) Density per band ~ PC scores ----------------------------------------
db_pca = db_all(vm_pca,:);

beta_log_density = nan(7, Nbands);
R2_density_rob   = nan(1, Nbands);
R2_density_ols   = nan(1, Nbands);
nOut_density     = nan(1, Nbands);

for b = 1:Nbands
    vmb = ~isnan(db_pca(:,b)) & db_pca(:,b) > 0;
    if sum(vmb) > 5
        Xd = X_pc(vmb,:);
        yd = log(db_pca(vmb, b));
        [bb, ~, r2r, r2o, no] = robustRegress(Xd, yd, irlsOpts);
        beta_log_density(:,b) = bb;
        R2_density_rob(b) = r2r;  R2_density_ols(b) = r2o;  nOut_density(b) = no;
    end
end
fprintf('log(Density) bands  R2_rob=[%s]\n', num2str(R2_density_rob,'%.3f '));

% --- 3d) Data-scale Nmodes range ---------------------------------------------
bw_data      = (bandEdges(:,2) - bandEdges(:,1))';
Nmodes_data  = nan(N_pca, 1);
for i = 1:N_pca
    row = db_pca(i,:);
    if ~any(isnan(row)), Nmodes_data(i) = sum(row .* bw_data); end
end
Nmodes_data_valid = Nmodes_data(~isnan(Nmodes_data));
q1n = prctile(Nmodes_data_valid,25); q3n = prctile(Nmodes_data_valid,75); iqrn = q3n-q1n;
Nmodes_data_inl  = Nmodes_data_valid(Nmodes_data_valid >= q1n-iqrn & Nmodes_data_valid <= q3n+iqrn);
Nmodes_data_range = [prctile(Nmodes_data_inl,5), prctile(Nmodes_data_inl,95)];
fprintf('Data-scale Nmodes range: [%.0f, %.0f]\n', Nmodes_data_range);

% --- 3e) Spacing CV range ----------------------------------------------------
scv_valid = scvAll(~isnan(scvAll));
q1 = prctile(scv_valid,25); q3 = prctile(scv_valid,75); iqr_v = q3-q1;
scv_inl  = scv_valid(scv_valid >= q1-iqr_v & scv_valid <= q3+iqr_v);
nOut_scv = numel(scv_valid) - numel(scv_inl);
diffusion_range = [prctile(scv_inl,5), prctile(scv_inl,95)];
diffusion_mean  = mean(scv_inl);
fprintf('Diffusion CV   range=[%.3f, %.3f]  outliers=%d\n', diffusion_range(1), diffusion_range(2), nOut_scv);

% --- 3f) FIR mean template ---------------------------------------------------
hFIR_list   = hFIR_all_c;
hFIR_ne     = ~cellfun(@isempty, hFIR_list);
hFIR_energy = zeros(numel(hFIR_list), 1);
hFIR_energy(hFIR_ne) = cellfun(@(h) sum(h.^2), hFIR_list(hFIR_ne), 'UniformOutput', true);

if any(hFIR_ne)
    eVec = hFIR_energy(hFIR_ne);
    eMed = median(eVec,'omitnan'); eMAD = mad(eVec,1);
    eFIR_inl = abs(eVec - eMed) <= 2.0*eMAD/0.6745;
    nOut_FIR = sum(~eFIR_inl);
    hFIR_list_filt = hFIR_list(hFIR_ne); hFIR_list_filt = hFIR_list_filt(eFIR_inl);
    Nfir = numel(hFIR_list_filt{1});
    hFIR_matrix = nan(numel(hFIR_list_filt), Nfir);
    for k = 1:numel(hFIR_list_filt)
        h = hFIR_list_filt{k};
        if numel(h)==Nfir, hFIR_matrix(k,:) = h(:)'/(max(abs(h))+eps); end
    end
    hFIR_matrix = hFIR_matrix(~any(isnan(hFIR_matrix),2),:);
    hFIR_mean = mean(hFIR_matrix,1)';
    hFIR_std  = std(hFIR_matrix,0,1)';
    fprintf('FIR template   %d taps from %d IRs (excluded %d energy outliers)\n', ...
        Nfir, size(hFIR_matrix,1), nOut_FIR);
else
    Nfir=256; hFIR_mean=zeros(Nfir,1); hFIR_mean(1)=1; hFIR_std=zeros(Nfir,1); nOut_FIR=0;
end

% --- 3g) C80 dataset range ---------------------------------------------------
C80_valid = C80v(isfinite(C80v));
q1c = prctile(C80_valid,25); q3c = prctile(C80_valid,75); iqr_c = q3c-q1c;
C80_inl   = C80_valid(C80_valid >= q1c-iqr_c & C80_valid <= q3c+iqr_c);
C80_range = [prctile(C80_inl,5), prctile(C80_inl,95)];
fprintf('C80_mid range  [%.2f, %.2f] dB\n', C80_range);

%% 4) Dataset ranges + PCA params
stats = struct();
stats.T60_range         = robustRange(T60v);
stats.Ts_range          = robustRange(Tsv);
stats.BR_range          = robustRange(BRv);
stats.TR_range          = robustRange(TRv);
stats.Nmodes_range      = robustRange(Nmodes_all);
stats.diffusion_range   = diffusion_range;
stats.diffusion_mean    = diffusion_mean;
stats.C80_range         = C80_range;
stats.amp_equip_scale   = amp_equip_scale;
stats.Nmodes_data_range = Nmodes_data_range;
stats.pca_mu            = pca_mu;
stats.pca_sigma         = pca_sigma;
stats.pca_coeff         = pca_coeff;
stats.pca_explained     = pca_explained;
stats.param_names       = {'T60','Ts','BR','TR','SpacingCV','C80'};

fprintf('\n--- Dataset ranges ---\n');
fprintf('T60:       [%.3f, %.3f] s\n',  stats.T60_range);
fprintf('Ts:        [%.4f, %.4f] s\n',  stats.Ts_range);
fprintf('BR:        [%.3f, %.3f]\n',    stats.BR_range);
fprintf('TR:        [%.3f, %.3f]\n',    stats.TR_range);
fprintf('Diffusion: [%.3f, %.3f]\n',    stats.diffusion_range);
fprintf('C80_mid:   [%.3f, %.3f] dB\n', stats.C80_range);

%% 5) Goodness-of-fit summary
fprintf('\n=== FIT QUALITY SUMMARY (per-band log(sigma) and log(rho), PC predictors) ===\n');
fprintf('%-26s  %8s  %8s  %10s\n','Regression','R2_OLS','R2_rob','nOutliers');
fprintf('%s\n', repmat('-',1,58));
for b = 1:Nbands
    printRow(sprintf('Sigma: %s', bandNames5{b}), R2_sigma_ols(b), R2_sigma_rob(b), nOut_sigma(b));
end
fprintf('%-26s  %8s  %8s  k=%.4f (std=%.4f)\n','Amp (equipartition)','-','-', amp_equip_scale, amp_equip_std);
for b = 1:Nbands
    printRow(sprintf('Density: %s', bandNames5{b}), R2_density_ols(b), R2_density_rob(b), nOut_density(b));
end

%% 6) Pack model
paramModel = struct();
paramModel.version             = 'v10-6param-perband';
paramModel.bandEdges           = bandEdges;
paramModel.bandCenters         = bandCenters;
paramModel.stats               = stats;
paramModel.beta_log_sigma      = beta_log_sigma;    % [7×5]: x_reg → log(sigma_b)
paramModel.beta_log_density    = beta_log_density;   % [7×5]: x_reg → log(density_b)
paramModel.hFIR_mean           = hFIR_mean;
paramModel.hFIR_std            = hFIR_std;
paramModel.fitQuality = struct( ...
    'R2_sigma_ols',    R2_sigma_ols,    'R2_sigma_robust',  R2_sigma_rob,  'nOut_sigma',  nOut_sigma, ...
    'amp_equip_scale', amp_equip_scale, 'amp_equip_std',    amp_equip_std, ...
    'R2_density_ols',  R2_density_ols,  'R2_density_robust', R2_density_rob, 'nOut_density', nOut_density, ...
    'nOut_diffusion', nOut_scv);

if ~isempty(saveFile)
    save(saveFile, 'paramModel', '-v7.3');
    fprintf('\nModel saved to %s\n', saveFile);
end
fprintf('\n=== MODEL BUILT (v10: per-band sigma and density, syn-only) ===\n');
end

%% Local helpers ---------------------------------------------------------------
function rng = robustRange(v)
    v = v(~isnan(v));
    q1=prctile(v,25); q3=prctile(v,75); iqr_v=q3-q1;
    v = v(v >= q1-1.0*iqr_v & v <= q3+1.0*iqr_v);
    rng = [prctile(v,5), prctile(v,95)];
end

function printRow(name, r2ols, r2rob, no)
    fprintf('%-24s  %8.4f  %8.4f  %10d\n', name, r2ols, r2rob, no);
end