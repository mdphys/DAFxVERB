function [aOut, bOut, cOut, fOut, hFIR_out] = parametric_reverb_generate(paramModel, userParams)
%PARAMETRIC_REVERB_GENERATE  Generate modal coefficients from 6 user parameters.
%
%   7 USER PARAMETERS (struct fields):
%     T60        : mid-freq reverberation time [s]          (default 1.0)
%     roomSize   : 0 (small) to 1 (large)                  (default 0.5)
%     warmth     : bass ratio BR, >1 = warm                 (default 1.0)
%     brightness : treble ratio TR, >1 = bright             (default 1.0)
%     diffusion  : 0 = metallic/regular, 1 = natural/random (default 0.7)
%     earlyLate  : 0 = late/diffuse tail, 1 = early/clear  (default 0.5)
%     fs         : sample rate [Hz]                         (default 44100)
%     fRange     : [fMin fMax] Hz                           (default [40 12000])
%     allowOOD   : if true, disables clamping to trained    (default false)
%                  ranges — use to generate out-of-distribution
%                  IRs (e.g. for t-SNE boundary validation)
%
%   earlyLate maps to C80 (mid-freq clarity dB), which enters the 6-parameter
%   PCA rotation.  hFIR energy is set by physics: E_early = E_late * 10^(C80/10).

%% 0) Defaults
def = struct('T60',1.0,'roomSize',0.5,'warmth',1.0,'brightness',1.0, ...
             'diffusion',0.7,'earlyLate',0.5,'fs',44100,'fRange',[40 12000], ...
             'allowOOD',false);
for fn = fieldnames(def)'
    if ~isfield(userParams, fn{1}), userParams.(fn{1}) = def.(fn{1}); end
end

T60  = userParams.T60;
rSz  = userParams.roomSize;
BR   = userParams.warmth;
TR   = userParams.brightness;
diff = userParams.diffusion;
el   = userParams.earlyLate;
fs   = userParams.fs;   %#ok
fMin = userParams.fRange(1);
fMax = userParams.fRange(2);
stats = paramModel.stats;

% Clamp to trained ranges (20% margin) — disabled when allowOOD=true
if ~userParams.allowOOD
    T60  = max(0.05, T60);   % floor only — anchor handles any T60
    BR   = clamp(BR,   stats.BR_range,  0.2);
    TR   = clamp(TR,   stats.TR_range,  0.2);
    diff = max(0, min(1, diff));
    el   = max(0, min(1, el));
end

Ts        = stats.Ts_range(1)        + rSz  * (stats.Ts_range(2)        - stats.Ts_range(1));
SpacingCV = stats.diffusion_range(1) + diff * (stats.diffusion_range(2)  - stats.diffusion_range(1));
C80       = stats.C80_range(1)       + el   * (stats.C80_range(2)        - stats.C80_range(1));

fprintf('Generating: T60=%.2f  size=%.2f  BR=%.2f  TR=%.2f  diff=%.2f  earlyLate=%.2f (C80=%.1fdB)\n', ...
    T60, rSz, BR, TR, diff, el, C80);

%% 0b) Orthogonalise: 6 user params -> orthogonal PC scores (6-dim PCA)
raw_params = [T60, Ts, BR, TR, SpacingCV, C80];
std_params = (raw_params - stats.pca_mu) ./ stats.pca_sigma;
pc_scores  = std_params * stats.pca_coeff;   % [1×6]
x_reg      = [1, pc_scores];                 % [1×7] design vector
fprintf('  PC scores: [%s]\n', sprintf('%.3f ', pc_scores));

%% 1) Number of modes
Nsynth_min = 1000;
Nsynth_max = 8000;

bandEdges = paramModel.bandEdges;
Nbands    = size(bandEdges, 1);
bd        = paramModel.beta_log_density;   % [7×Nbands]: x_reg -> log(density_b)
bw        = bandEdges(:,2) - bandEdges(:,1);

density_pred = nan(Nbands, 1);
for ib = 1:Nbands
    if ~any(isnan(bd(:,ib)))
        density_pred(ib) = exp(x_reg * bd(:,ib));   % guaranteed positive
    else
        density_pred(ib) = 1.0;
    end
end

Nmodes_raw = sum(density_pred .* bw);
dr = stats.Nmodes_data_range;
t  = max(0, min(1, (Nmodes_raw - dr(1)) / max(dr(2)-dr(1), eps)));
Nmodes = round(Nsynth_min + t*(Nsynth_max - Nsynth_min));
fprintf('  Nmodes_raw=%.0f  t=%.3f  ->  Nmodes=%d\n', Nmodes_raw, t, Nmodes);

%% 2) Modal frequencies
mpb_float = density_pred .* bw;
mpb_float = mpb_float / sum(mpb_float) * Nmodes;
mpb       = round(mpb_float);
dn        = Nmodes - sum(mpb);
[~,si]    = sort(mpb_float - mpb, 'descend');
for k = 1:abs(dn), mpb(si(k)) = mpb(si(k)) + sign(dn); end

jitterFrac = diff * 0.9;
fOut = [];
for ib = 1:Nbands
    Nb = mpb(ib);
    if Nb <= 0, continue; end
    fLo  = max(fMin, bandEdges(ib,1));
    fHi  = min(fMax, bandEdges(ib,2));
    fReg = linspace(fLo, fHi, Nb+2); fReg = fReg(2:end-1);
    spacing = mean(diff_vec(fReg));
    jitter  = (rand(1,Nb)-0.5)*spacing*jitterFrac;
    fOut    = [fOut, max(fLo, min(fHi, fReg+jitter))]; %#ok
end
fOut = sort(fOut(:));
Nact = numel(fOut);

%% 3) Damping profile sigma(f) — per-band prediction + interpolation
%   Predict log(sigma_b) at 5 band centres, then interpolate to per-mode
%   frequencies via pchip in log-frequency space.  Guaranteed positive.

bls = paramModel.beta_log_sigma;         % [7×5]
bc  = paramModel.bandCenters(:);         % [5×1] geometric band centres

sigma_band = nan(Nbands, 1);
for ib = 1:Nbands
    if ~any(isnan(bls(:,ib)))
        sigma_band(ib) = exp(x_reg * bls(:,ib));
    else
        sigma_band(ib) = 1.0;
    end
end

% Interpolate to per-mode frequencies (pchip on log10(f), log(sigma))
cOut = exp(interp1(log10(bc), log(sigma_band), log10(fOut), 'linear', 'extrap'));

% Physics anchor: sigma_mid = 6.908/T60
sigma_mid_target = 6.908 / T60;
midMask = fOut >= 500 & fOut <= 1000;
if any(midMask)
    sigma_mid_actual = mean(cOut(midMask));
else
    sigma_mid_actual = mean(cOut);
end
cOut = cOut * (sigma_mid_target / (sigma_mid_actual + eps));

% BR/TR shape correction
for i = 1:Nact
    fi = fOut(i);
    if fi < 500
        w = max(0,min(1, 1-log2(fi/40)/log2(500/40)));
        cOut(i) = cOut(i)*(1 + w*(1/BR - 1));
    elseif fi > 1000
        w = max(0,min(1, log2(fi/1000)/log2(12000/1000)));
        cOut(i) = cOut(i)*(1 + w*(1/TR - 1));
    end
end
cOut = max(cOut, 0.1);

%% 4) Amplitude: equipartition amp(f) = k * sqrt(sigma(f))
k_equip = stats.amp_equip_scale;
amp_out = k_equip * sqrt(cOut);
amp_out = amp_out / sqrt(sum(amp_out.^2)) * sqrt(Nact);

phase = 2*pi*rand(Nact,1);
aOut  = amp_out .* cos(phase);
bOut  = amp_out .* sin(phase);

%% 5) hFIR: scale by physics-derived early energy from C80
%
%   C80 = 10*log10(E_early / E_late)  [dB]
%   => E_early = E_late * 10^(C80/10)
%
%   E_late is the total modal tail energy: sum(amp_i^2 / (2*sigma_i)).
%   hFIR_mean is then scaled so sum(hFIR^2) = E_early.
%
%   Pure physics formula — no regression needed.
%   earlyLate=0 -> C80=C80_min -> small early energy (late, diffuse)
%   earlyLate=1 -> C80=C80_max -> large early energy (clear, present)

E_late   = sum(amp_out.^2 ./ (2 * cOut + eps));   % total modal tail energy
E_early  = E_late * 10^(C80 / 10);                % C80 definition
hFIR_raw   = paramModel.hFIR_mean;
e_template = sum(hFIR_raw.^2) + eps;
hFIR_raw   = hFIR_raw * sqrt(E_early / e_template);

if     numel(hFIR_raw) < 256, hFIR_out = [hFIR_raw; zeros(256-numel(hFIR_raw),1)];
elseif numel(hFIR_raw) > 256, hFIR_out = hFIR_raw(1:256);
else,                          hFIR_out = hFIR_raw; end

fprintf('  %d modes  sigma=[%.2f, %.2f]  k=%.4f  hFIR_E=%.4f\n', ...
    Nact, min(cOut), max(cOut), k_equip, sum(hFIR_out.^2));
end

function v = clamp(v, range, margin)
    m = (range(2)-range(1))*margin;
    v = max(range(1)-m, min(range(2)+m, v));
end
function d = diff_vec(x)
    d = x(2:end)-x(1:end-1);
    if isempty(d), d=1; end
end