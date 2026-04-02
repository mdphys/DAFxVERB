function [beta, inlierMask, R2, R2_ols, nOutliers] = robustRegress(X, y, opts)
%ROBUSTREGRESS  IRLS with bisquare (Tukey) weights for outlier-robust regression.
%
%   [beta, inlierMask, R2, R2_ols, nOutliers] = robustRegress(X, y)
%   [beta, inlierMask, R2, R2_ols, nOutliers] = robustRegress(X, y, opts)
%
%   Two-stage pipeline:
%     1) Hard MAD pre-screen  — removes gross outliers (|r_OLS| > k*sigma_MAD)
%        before IRLS sees them, preventing leverage-point masking.
%     2) IRLS bisquare        — iteratively downweights remaining soft outliers.
%
%   INPUTS
%     X   [n x p]  Design matrix (include a ones column for intercept)
%     y   [n x 1]  Response vector
%     opts         Optional struct:
%       .c            (default 3.0)   Bisquare tuning constant.
%                                     4.685 = 95% Gaussian efficiency (lenient).
%                                     3.0   = more aggressive (recommended).
%                                     2.0   = very aggressive.
%       .prescreen    (default true)  Enable hard MAD pre-screen pass.
%       .prescreen_k  (default 3.0)   MAD multiplier for pre-screen cutoff.
%       .maxIter      (default 50)    Max IRLS iterations.
%       .tol          (default 1e-6)  Convergence tolerance on beta.
%       .verbose      (default false) Print iteration info.
%
%   OUTPUTS
%     beta        [p x 1]  Robust coefficient vector
%     inlierMask  [n x 1]  logical — true = inlier (nonzero bisquare weight
%                           AND passed pre-screen)
%     R2          scalar   R² on inliers only
%     R2_ols      scalar   R² of initial OLS fit (for comparison)
%     nOutliers   scalar   Total outliers (pre-screen + IRLS combined)

if nargin < 3, opts = struct(); end
maxIter    = getfield_default(opts, 'maxIter',    50);
tol        = getfield_default(opts, 'tol',        1e-6);
c          = getfield_default(opts, 'c',          3.0);    % aggressive: was 4.685
prescreen  = getfield_default(opts, 'prescreen',  true);   % hard MAD pre-screen
prescreen_k= getfield_default(opts, 'prescreen_k',3.0);    % MAD multiplier for pre-screen
verbose    = getfield_default(opts, 'verbose',    false);

[n, p] = size(X);

% --- Initial OLS fit -------------------------------------------------------
beta_ols = X \ y;
R2_ols   = calcR2(y, X * beta_ols);

% --- Hard MAD pre-screen (catches gross outliers before IRLS) --------------
% Removes points whose OLS residual exceeds prescreen_k * MAD-sigma.
% This helps IRLS converge faster and avoids masking effects from extreme
% leverage points that can still bias the initial iterations.
prescreenMask = true(n, 1);
if prescreen
    r_ols = y - X * beta_ols;
    s_ols = mad(r_ols, 1) / 0.6745;
    if s_ols > eps
        prescreenMask = abs(r_ols) <= prescreen_k * s_ols;
    end
    nPre = sum(~prescreenMask);
    if verbose && nPre > 0
        fprintf('  Pre-screen removed %d / %d points (%.1f%%)\n', nPre, n, 100*nPre/n);
    end
end

% Subset to pre-screen inliers for IRLS
Xp = X(prescreenMask, :);
yp = y(prescreenMask);
np = sum(prescreenMask);

% --- IRLS on pre-screened data ---------------------------------------------
beta = Xp \ yp;   % warm start from OLS on clean subset
W    = ones(np, 1);

for iter = 1:maxIter
    beta_prev = beta;

    % Guard: if all weights are zero (every point flagged as outlier),
    % the weighted normal equations are singular.  Fall back to the last
    % valid beta rather than crashing.
    if all(W == 0)
        if verbose
            fprintf('  IRLS iter %2d: all weights zero — keeping previous beta\n', iter);
        end
        beta = beta_prev;
        break;
    end

    Xw   = Xp .* W;
    yw   = yp .* W;

    % Guard: check conditioning before solving
    A = Xw' * Xw;
    if rcond(A) < eps
        if verbose
            fprintf('  IRLS iter %2d: singular normal equations (rcond=%.2e) — keeping previous beta\n', iter, rcond(A));
        end
        beta = beta_prev;
        break;
    end
    beta = A \ (Xw' * yw);

    r = yp - Xp * beta;
    s = mad(r, 1) / 0.6745;

    if s < eps, break; end

    u = r / (c * s);
    W = ((1 - u.^2).^2) .* (abs(u) < 1);

    delta = norm(beta - beta_prev) / (norm(beta_prev) + eps);
    if verbose
        fprintf('  IRLS iter %2d: delta=%.2e, nZeroW=%d\n', iter, delta, sum(W==0));
    end
    if delta < tol, break; end
end

% --- Final outputs ----------------------------------------------------------
% Build full-length inlier mask: pre-screen failures are always outliers
inlierMask = false(n, 1);
inlierMask(prescreenMask) = W > 0;
nOutliers  = sum(~inlierMask);

yi    = y(inlierMask);
yhati = (X(inlierMask,:) * beta);
R2    = calcR2(yi, yhati);

end

% ---------------------------------------------------------------------------
function v = getfield_default(s, f, default)
    if isfield(s, f)
        v = s.(f);
    else
        v = default;
    end
end

function R2 = calcR2(y, yhat)
    ss_res = sum((y - yhat).^2);
    ss_tot = sum((y - mean(y)).^2);
    R2 = 1 - ss_res / (ss_tot + eps);
end
