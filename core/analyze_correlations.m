function modelData = analyze_correlations(featureTable, savePlots)
%ANALYZE_CORRELATIONS  Correlation analysis and regression diagnostics.
%
%   v6: Full-matrix PCA variance report added.
%       Orthogonality figure: 1x3 (Pearson only, no Spearman).
%       All fonts enlarged for paper readability.

if nargin < 2, savePlots = false; end
fprintf('\n========== CORRELATION ANALYSIS ==========\n');

%% 1) Numeric feature matrix
varNames = featureTable.Properties.VariableNames;
numericVars = {};
for i = 1:numel(varNames)
    v = varNames{i};
    if strcmp(v,'filename'), continue; end
    if isnumeric(featureTable.(v)), numericVars{end+1} = v; end
end

X = table2array(featureTable(:, numericVars));
[Nsamples, Nfeats] = size(X);
fprintf('Feature matrix: %d samples x %d features\n', Nsamples, Nfeats);

% NaN imputation
for j = 1:Nfeats
    col = X(:,j); nm = isnan(col);
    if any(nm), col(nm) = median(col(~nm)); X(:,j) = col; end
end
mu = mean(X,1); sd = std(X,0,1); sd(sd<eps)=1;

%% 2) Full-matrix PCA — how many components for various variance thresholds
goodCols = std(X,0,1) > 1e-10;
Xz = (X(:,goodCols) - mean(X(:,goodCols),1)) ./ std(X(:,goodCols),0,1);
[~, ~, ~, ~, explained_full] = pca(Xz);
cumvar = cumsum(explained_full);

fprintf('\n--- Full feature matrix PCA (%d features, %d constant dropped) ---\n', ...
    sum(goodCols), sum(~goodCols));
thresholds = [50 70 80 90 95 99];
for th = thresholds
    npc = find(cumvar >= th, 1);
    fprintf('  %2d%% variance: %d components\n', th, npc);
end
%% 3) Correlation matrix figure
R = corrcoef(X);
D = pdist(R','correlation');
Zc = linkage(D,'average');
leafOrder = optimalleaforder(Zc,D);

figure('Position',[50 50 1000 900]);
imagesc(R(leafOrder,leafOrder)); colorbar;
cmap = bluewhitered_cmap(256); colormap(cmap); caxis([-1 1]);
set(gca,'XTick',1:Nfeats,'XTickLabel',numericVars(leafOrder), ...
        'YTick',1:Nfeats,'YTickLabel',numericVars(leafOrder), ...
        'TickLabelInterpreter','none');
xtickangle(45); set(gca,'FontSize',7);
title('Pearson Correlation Matrix (hierarchically reordered)','FontSize',13);
if savePlots, exportgraphics(gcf,'corr_matrix.png','Resolution',200); end

%% 4) 6-parameter PCA + IRLS regressions
bandCols  = {'sigma_subBass','sigma_bass','sigma_mid','sigma_presence','sigma_brilliance'};
densCols  = {'density_subBass','density_bass','density_mid','density_presence','density_brilliance'};
bandTitls = {'subBass','bass','mid','presence','brilliance'};

paramVars6 = {'T30_mean_mid','Ts','BR','TR','spacing_cv','C80_mean_mid'};
irlsOpts   = struct('maxIter',50,'tol',1e-6,'c',3.0,'prescreen',true,'prescreen_k',3.0,'verbose',false);

P6 = nan(height(featureTable), 6);
for k=1:6
    if ismember(paramVars6{k},numericVars), P6(:,k) = featureTable.(paramVars6{k}); end
end
rowOK5 = all(isfinite(P6),2);
P5     = P6(rowOK5,:);
N5_reg = sum(rowOK5);
fprintf('Complete cases for PC regressions: %d\n', N5_reg);

p5_mu    = median(P5,1);
p5_sigma = (prctile(P5,75) - prctile(P5,25)) / 1.349;
p5_sigma(p5_sigma < eps) = 1;
P5_std   = (P5 - p5_mu) ./ p5_sigma;

[p5_coeff, p5_scores, ~, ~, p5_explained] = pca(P5_std);
X_pc5 = [ones(N5_reg,1), p5_scores];

fprintf('6-param PC variance: '); fprintf('%.1f%%  ',p5_explained); fprintf('\n');

% Per-band sigma regressions
sb_all = nan(height(featureTable),5);
for b=1:5
    if ismember(bandCols{b},numericVars), sb_all(:,b) = featureTable.(bandCols{b}); end
end
sb_all = sb_all(rowOK5,:);

beta_sb = nan(7,5); R2_sb_rob = nan(1,5); R2_sb_ols = nan(1,5);
nOut_sb = nan(1,5); yhat_sb = nan(N5_reg,5); inl_sb = true(N5_reg,5);
for b=1:5
    vmb = ~isnan(sb_all(:,b)) & sb_all(:,b) > 0;
    if sum(vmb)>5
        [bb,inl_b,r2r,r2o,no] = robustRegress(X_pc5(vmb,:), log(sb_all(vmb,b)), irlsOpts);
        beta_sb(:,b)=bb; R2_sb_rob(b)=r2r; R2_sb_ols(b)=r2o; nOut_sb(b)=no;
        inl_sb(vmb,b)=inl_b; yhat_sb(vmb,b) = X_pc5(vmb,:)*bb;
    end
end
fprintf('Per-band log(sigma)  R2_rob=[%s]\n', num2str(R2_sb_rob,'%.3f '));

% Per-band density regressions
db_all = nan(height(featureTable),5);
for b=1:5
    if ismember(densCols{b},numericVars), db_all(:,b) = featureTable.(densCols{b}); end
end
db_all = db_all(rowOK5,:);

beta_db = nan(7,5); R2_db_rob = nan(1,5); R2_db_ols = nan(1,5);
nOut_db = nan(1,5); yhat_db = nan(N5_reg,5); inl_db = true(N5_reg,5);
for b=1:5
    vmb_pos = ~isnan(db_all(:,b)) & db_all(:,b) > 0;
    if sum(vmb_pos)>5
        [bb,inl_b,r2r,r2o,no] = robustRegress(X_pc5(vmb_pos,:), log(db_all(vmb_pos,b)), irlsOpts);
        beta_db(:,b)=bb; R2_db_rob(b)=r2r; R2_db_ols(b)=r2o; nOut_db(b)=no;
        inl_db(vmb_pos,b)=inl_b; yhat_db(vmb_pos,b) = X_pc5(vmb_pos,:)*bb;
    end
end
fprintf('Per-band log(density) R2_rob=[%s]\n', num2str(R2_db_rob,'%.3f '));

%% 5) param_orthogonality.png — 1x3, Pearson only
paramVars  = {'T30_mean_mid','Ts','BR','TR','spacing_cv','C80_mean_mid'};
paramLabels= {'T_{30}','T_s','BR','TR','CV_\Delta','C_{80}'};
Np5 = numel(paramVars);

Pmat = nan(height(featureTable), Np5);
for k = 1:Np5
    if ismember(paramVars{k}, numericVars)
        Pmat(:,k) = featureTable.(paramVars{k});
    end
end
rowOK = all(isfinite(Pmat), 2);
Pmat  = Pmat(rowOK, :);
N5    = sum(rowOK);
fprintf('\n--- PARAMETER ORTHOGONALITY (N=%d IRs) ---\n', N5);

Rp = nan(Np5); Pp = nan(Np5);
for i = 1:Np5
    for j = 1:Np5
        [rp, pp] = corr(Pmat(:,i), Pmat(:,j), 'Type','Pearson');
        Rp(i,j) = rp; Pp(i,j) = pp;
    end
end

cmapBWR = bluewhitered_cmap(256);
nPCs = numel(p5_explained);
pcLbls = arrayfun(@(k)sprintf('PC%d',k), 1:nPCs, 'UniformOutput', false);

figO = figure('Position',[60 60 1800 520], 'Name','Parameter Orthogonality');
sgtitle(sprintf('6-Parameter Orthogonality  (N=%d IRs)', N5), ...
    'FontSize',15,'FontWeight','bold');

% [1] Pearson r
subplot(1,3,1);
imagesc(Rp); colormap(cmapBWR); caxis([-1 1]); colorbar;
set(gca,'XTick',1:Np5,'XTickLabel',paramLabels,'XTickLabelRotation',30,'FontSize',12, ...
        'YTick',1:Np5,'YTickLabel',paramLabels);
title('Pearson  r','FontSize',14);
for i=1:Np5, for j=1:Np5
    if i~=j
        sig=''; if Pp(i,j)<0.001,sig='***'; elseif Pp(i,j)<0.01,sig='**'; elseif Pp(i,j)<0.05,sig='*'; end
        clr=[1 1 1]*double(abs(Rp(i,j))>0.4);
        text(j,i,sprintf('%.2f%s',Rp(i,j),sig),'HorizontalAlignment','center','FontSize',11,'Color',clr);
    else
        text(j,i,'\textemdash','HorizontalAlignment','center','FontSize',12, ...
            'Color',[0.45 0.45 0.45],'Interpreter','latex');
    end
end; end

% [2] PCA scree
subplot(1,3,2);
bar(1:nPCs, p5_explained, 'FaceColor',[0.2 0.5 0.85],'EdgeColor','none'); hold on;
plot(1:nPCs, cumsum(p5_explained), 'ro-', 'LineWidth',2.0, 'MarkerSize',8, 'MarkerFaceColor','r');
yline(95,'k--','LineWidth',1,'Label','95%','LabelHorizontalAlignment','left','FontSize',11);
hold off; grid on;
set(gca,'XTick',1:nPCs,'XTickLabel',pcLbls,'FontSize',12);
ylabel('Variance explained (%)','FontSize',12);
title('PCA scree  (6-param subspace)','FontSize',14);
ylim([0 110]);

% [3] PC score correlations
subplot(1,3,3);
Pstd_v = (Pmat - median(Pmat)) ./ max((prctile(Pmat,75)-prctile(Pmat,25))/1.349, eps);
[~, pc_v] = pca(Pstd_v);
Rpc = corr(pc_v, 'Type','Pearson');
imagesc(Rpc); colormap(cmapBWR); caxis([-1 1]); colorbar;
set(gca,'XTick',1:Np5,'XTickLabel',pcLbls,'FontSize',12, ...
        'YTick',1:Np5,'YTickLabel',pcLbls);
title('PC score correlations','FontSize',14);
off_rpc = Rpc - eye(6);
for i=1:Np5, for j=1:Np5
    if i~=j
        text(j,i,sprintf('%.0e',Rpc(i,j)),'HorizontalAlignment','center','FontSize',10,'Color',[1 1 1]);
    else
        text(j,i,'1','HorizontalAlignment','center','FontSize',11,'Color',[0.35 0.35 0.35]);
    end
end; end
text(0.5,-0.08,sprintf('max|off-diag| = %.1e',max(abs(off_rpc(:)))),'Units','normalized', ...
    'HorizontalAlignment','center','FontSize',11,'Color',[0 0.5 0]);

if savePlots, exportgraphics(figO,'param_orthogonality.png','Resolution',200); end

%% 6) fit_quality.png — 4x5, larger fonts
figure('Position',[30 30 1800 1200]);
sgtitle(sprintf('Fit Quality: \\sigma and \\rho per Band  (N=%d IRs)', N5_reg), ...
    'FontSize',15,'FontWeight','bold');
clrs5 = lines(5);
xr = linspace(-4,4,100);

for b = 1:5
    subplot(4,5,b);
    vmb = ~isnan(sb_all(:,b)) & sb_all(:,b) > 0;
    plotPredActual_ac(log(sb_all(vmb,b)), yhat_sb(vmb,b), inl_sb(vmb,b), ...
        R2_sb_rob(b), R2_sb_ols(b), nOut_sb(b), ...
        sprintf('log(\\sigma_{%s})', bandTitls{b}));

    subplot(4,5,5+b);
    ry = log(sb_all(vmb,b)); rh = yhat_sb(vmb,b); inlb = inl_sb(vmb,b);
    res = (ry(inlb) - rh(inlb)) / (std(ry(inlb)) + eps);
    histogram(res, 30, 'Normalization','pdf', ...
        'FaceColor',clrs5(b,:),'EdgeColor','none','FaceAlpha',0.75); hold on;
    plot(xr, normpdf(xr,0,1), 'k--', 'LineWidth',1.5);
    hold off; grid on; xlim([-4 4]);
    xlabel('Norm. residual','FontSize',10); ylabel('PDF','FontSize',10);
    title(sprintf('\\sigma_{%s}  residuals', bandTitls{b}),'FontSize',11);

    subplot(4,5,10+b);
    vmb2 = ~isnan(db_all(:,b)) & db_all(:,b) > 0;
    plotPredActual_ac(log(db_all(vmb2,b)), yhat_db(vmb2,b), inl_db(vmb2,b), ...
        R2_db_rob(b), R2_db_ols(b), nOut_db(b), ...
        sprintf('log(\\rho_{%s})', bandTitls{b}));

    subplot(4,5,15+b);
    ry2 = log(db_all(vmb2,b)); rh2 = yhat_db(vmb2,b); inlb2 = inl_db(vmb2,b);
    res2 = (ry2(inlb2) - rh2(inlb2)) / (std(ry2(inlb2)) + eps);
    histogram(res2, 30, 'Normalization','pdf', ...
        'FaceColor',clrs5(b,:)*0.7+0.3,'EdgeColor','none','FaceAlpha',0.75); hold on;
    plot(xr, normpdf(xr,0,1), 'k--', 'LineWidth',1.5);
    hold off; grid on; xlim([-4 4]);
    xlabel('Norm. residual','FontSize',10); ylabel('PDF','FontSize',10);
    title(sprintf('\\rho_{%s}  residuals', bandTitls{b}),'FontSize',11);
end

annotation('textbox',[0.01 0.76 0.03 0.1],'String','\sigma fit','FontSize',12, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');
annotation('textbox',[0.01 0.52 0.03 0.1],'String','\sigma resid.','FontSize',12, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');
annotation('textbox',[0.01 0.28 0.03 0.1],'String','\rho fit','FontSize',12, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');
annotation('textbox',[0.01 0.04 0.03 0.1],'String','\rho resid.','FontSize',12, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');

if savePlots, exportgraphics(gcf,'fit_quality.png','Resolution',200); end

fprintf('\n=== FIT QUALITY SUMMARY ===\n');
fprintf('%-30s  %8s  %8s  %10s\n','Regression','R2_OLS','R2_rob','nOut');
fprintf('%s\n',repmat('-',1,62));
for b=1:5
    fprintf('  sigma %-12s  %8.4f  %8.4f  %10d\n', bandTitls{b}, R2_sb_ols(b), R2_sb_rob(b), nOut_sb(b));
end
for b=1:5
    fprintf('  density %-10s  %8.4f  %8.4f  %10d\n', bandTitls{b}, R2_db_ols(b), R2_db_rob(b), nOut_db(b));
end

%% 6b) fit_scatter.png — 2x5, scatter only, no residuals
figure('Position',[30 30 1800 650]);
sgtitle(sprintf('Predicted vs Actual: \\sigma and \\rho per Band  (N=%d IRs)', N5_reg), ...
    'FontSize',15,'FontWeight','bold');

for b = 1:5
    % Row 1: sigma
    subplot(2,5,b);
    vmb = ~isnan(sb_all(:,b)) & sb_all(:,b) > 0;
    plotPredActual_ac(log(sb_all(vmb,b)), yhat_sb(vmb,b), inl_sb(vmb,b), ...
        R2_sb_rob(b), R2_sb_ols(b), nOut_sb(b), ...
        sprintf('log(\\sigma_{%s})', bandTitls{b}));

    % Row 2: rho
    subplot(2,5,5+b);
    vmb2 = ~isnan(db_all(:,b)) & db_all(:,b) > 0;
    plotPredActual_ac(log(db_all(vmb2,b)), yhat_db(vmb2,b), inl_db(vmb2,b), ...
        R2_db_rob(b), R2_db_ols(b), nOut_db(b), ...
        sprintf('log(\\rho_{%s})', bandTitls{b}));
end

annotation('textbox',[0.01 0.55 0.03 0.2],'String','Damping','FontSize',13, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');
annotation('textbox',[0.01 0.10 0.03 0.2],'String','Density','FontSize',13, ...
    'FontWeight','bold','EdgeColor','none','Rotation',90,'HorizontalAlignment','center');

if savePlots, exportgraphics(gcf,'fit_scatter.png','Resolution',200); end

%% 7) param_distributions.png
pctiles   = [5 25 50 75 95];
pct_clrs  = {[0.8 0.2 0.2],[0.9 0.6 0.1],[0.1 0.55 0.1],[0.9 0.6 0.1],[0.8 0.2 0.2]};
pct_styl  = {'--',':','-',':','--'};
pct_labs  = {'p5','p25','med','p75','p95'};
paramShort = {'T30','Ts','BR','TR','SpCV','C80'};
paramUnits = {'s', 's', '', '', '', 'dB'};

figPD = figure('Position',[50 50 1800 750], 'Name','Parameter Distributions');
sgtitle(sprintf('6-Parameter Corpus Distributions  (N=%d IRs)', N5), ...
    'FontSize',15, 'FontWeight','bold');

for k = 1:Np5
    x = Pmat(:,k); x = x(isfinite(x));
    pv = prctile(x, pctiles);

    subplot(2, Np5, k);
    histogram(x, 40, 'Normalization','probability', ...
        'FaceColor',[0.25 0.55 0.85], 'EdgeColor','none', 'FaceAlpha',0.75);
    hold on; yl = ylim;
    for pi = 1:numel(pctiles)
        xline(pv(pi), pct_styl{pi}, 'Color', pct_clrs{pi}, 'LineWidth', 1.4);
        text(pv(pi), yl(2)*0.97, pct_labs{pi}, 'FontSize',9, 'Color', pct_clrs{pi}, ...
            'HorizontalAlignment','center', 'VerticalAlignment','top');
    end
    hold off; grid on; box on;
    if ~isempty(paramUnits{k})
        xlabel(sprintf('%s  (%s)', paramShort{k}, paramUnits{k}), 'FontSize',12);
    else
        xlabel(paramShort{k}, 'FontSize',12);
    end
    ylabel('Proportion', 'FontSize',11);
    title(paramLabels{k}, 'FontSize',13, 'FontWeight','bold');
    ann = sprintf('p5  = %.3g\np25 = %.3g\nmed = %.3g\np75 = %.3g\np95 = %.3g', pv);
    text(0.97, 0.97, ann, 'Units','normalized', 'FontSize',9, ...
        'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'Color',[0.3 0.3 0.3], 'BackgroundColor','w', 'Margin',2, 'FontName','FixedWidth');

    subplot(2, Np5, Np5 + k);
    nPts = numel(x);
    jitter = (rand(nPts,1) - 0.5) * 0.35;
    scatter(jitter, x, 4, [0.5 0.7 0.9], 'filled', 'MarkerFaceAlpha', 0.15); hold on;
    q1 = pv(2); q3 = pv(4); med_v = pv(3);
    whislo = max(min(x), q1 - 1.5*(q3-q1));
    whishi = min(max(x), q3 + 1.5*(q3-q1));
    fill([-0.28 0.28 0.28 -0.28], [q1 q1 q3 q3], [0.25 0.55 0.85], ...
        'FaceAlpha',0.35, 'EdgeColor',[0.1 0.35 0.65], 'LineWidth',1.2);
    plot([-0.28 0.28], [med_v med_v], '-', 'Color',[0.05 0.3 0.6], 'LineWidth',2.2);
    plot([0 0], [whislo q1], '-', 'Color',[0.3 0.3 0.3], 'LineWidth',1.0);
    plot([0 0], [q3 whishi], '-', 'Color',[0.3 0.3 0.3], 'LineWidth',1.0);
    plot([-0.12 0.12], [whislo whislo], '-', 'Color',[0.3 0.3 0.3], 'LineWidth',1.0);
    plot([-0.12 0.12], [whishi whishi], '-', 'Color',[0.3 0.3 0.3], 'LineWidth',1.0);
    out_pts = x(x < whislo | x > whishi);
    if ~isempty(out_pts)
        scatter(zeros(numel(out_pts),1), out_pts, 8, [0.8 0.2 0.2], 'filled', 'MarkerFaceAlpha',0.5);
    end
    plot([-0.35 0.35], [pv(1) pv(1)], '--', 'Color',pct_clrs{1}, 'LineWidth',1.0);
    plot([-0.35 0.35], [pv(5) pv(5)], '--', 'Color',pct_clrs{5}, 'LineWidth',1.0);
    hold off; grid on; box on;
    xlim([-0.5 0.5]); set(gca,'XTick',[], 'FontSize',11);
    if ~isempty(paramUnits{k})
        ylabel(sprintf('%s  (%s)', paramShort{k}, paramUnits{k}), 'FontSize',12);
    else
        ylabel(paramShort{k}, 'FontSize',12);
    end
    title(sprintf('IQR = %.3g,  range = [%.3g, %.3g]', q3-q1, min(x), max(x)), 'FontSize',11);
end

if savePlots, exportgraphics(figPD, 'param_distributions.png', 'Resolution',200); end

%% 7b) param_histograms.png — clean 2x3, zoomed to meaningful range
figPH = figure('Position',[50 50 1400 700], 'Name','Parameter Histograms');

for k = 1:Np5
    x = Pmat(:,k); x = x(isfinite(x));
    pv = prctile(x, pctiles);

    % Zoom to [p1, p99] — kill the absurd outlier tails
    p1 = prctile(x, 1); p99 = prctile(x, 99);
    pad = 0.05*(p99 - p1);
    xClip = x(x >= p1 & x <= p99);

    subplot(2, 3, k);
    histogram(xClip, 40, 'Normalization','probability', ...
        'FaceColor',[0.25 0.55 0.85], 'EdgeColor','none', 'FaceAlpha',0.75);
    hold on; yl = ylim;
    for pi = 1:numel(pctiles)
        if pv(pi) >= p1-pad && pv(pi) <= p99+pad
            xline(pv(pi), pct_styl{pi}, 'Color', pct_clrs{pi}, 'LineWidth', 1.6);
            text(pv(pi), yl(2)*0.95, pct_labs{pi}, 'FontSize',11, 'Color', pct_clrs{pi}, ...
                'HorizontalAlignment','center', 'VerticalAlignment','top');
        end
    end
    hold off; grid on; box on;
    xlim([p1-pad, p99+pad]);
    if ~isempty(paramUnits{k})
        xlabel(sprintf('%s  (%s)', paramShort{k}, paramUnits{k}), 'FontSize',13);
    else
        xlabel(paramShort{k}, 'FontSize',13);
    end
    ylabel('Proportion', 'FontSize',12);
    title(paramLabels{k}, 'FontSize',14, 'FontWeight','bold');
    set(gca, 'FontSize',11);

    ann = sprintf('p5  = %.3g\np25 = %.3g\nmed = %.3g\np75 = %.3g\np95 = %.3g', pv);
    text(0.97, 0.97, ann, 'Units','normalized', 'FontSize',10, ...
        'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'Color',[0.3 0.3 0.3], 'BackgroundColor','w', 'Margin',2, 'FontName','FixedWidth');
end

if savePlots, exportgraphics(figPH, 'param_histograms.png', 'Resolution',200); end

%% 8) Pack output
regressions = struct('sigma_R2_rob',R2_sb_rob,'sigma_R2_ols',R2_sb_ols, ...
                     'density_R2_rob',R2_db_rob,'density_R2_ols',R2_db_ols, ...
                     'beta_sigma_band',beta_sb,'beta_density_band',beta_db, ...
                     'pca_coeff',p5_coeff,'pca_mu',p5_mu,'pca_sigma',p5_sigma, ...
                     'pca_explained',p5_explained, ...
                     'full_pca_explained',explained_full);
modelData = struct('numericVars',{numericVars},'mu',mu,'sd',sd,'R',R, ...
    'regressions',regressions,'featureTable',featureTable, ...
    'param_Rpearson', Rp, 'paramVars', {paramVars});
fprintf('\nAnalysis complete.\n');
end

%% ============================================================
function plotPredActual_ac(y, yhat, inl, R2_rob, R2_ols, nOut, titleStr)
    y=y(:); yhat=yhat(:); inl=logical(inl(:));
    if any(inl)
        scatter(yhat(inl), y(inl), 12, [0.2 0.5 0.85],'filled','MarkerFaceAlpha',0.4); hold on;
    end
    if any(~inl)
        scatter(yhat(~inl), y(~inl), 12, [0.7 0.7 0.7],'x','LineWidth',0.8);
    end
    allv=[y;yhat]; lo=min(allv); hi=max(allv); pad=(hi-lo)*0.05;
    plot([lo-pad,hi+pad],[lo-pad,hi+pad],'r-','LineWidth',1.8);
    if sum(inl)>2
        p=polyfit(yhat(inl),y(inl),1); xtr=[lo-pad,hi+pad];
        plot(xtr,polyval(p,xtr),'k--','LineWidth',1.0);
    end
    hold off; grid on; axis tight;
    xlabel('Predicted','FontSize',10); ylabel('Actual','FontSize',10);
    title(titleStr,'FontSize',11,'Interpreter','tex');
    nInl=sum(inl);
    text(0.04,0.97,sprintf('R^2_{rob}=%.3f',R2_rob),'Units','normalized','FontSize',10, ...
        'Color',[0.15 0.4 0.75],'FontWeight','bold','VerticalAlignment','top');
    text(0.04,0.86,sprintf('R^2_{ols}=%.3f',R2_ols),'Units','normalized','FontSize',10, ...
        'Color',[0.5 0.5 0.5],'VerticalAlignment','top');
    text(0.96,0.06,sprintf('n=%d  out=%d (%.0f%%)',nInl,nOut,100*nOut/(nInl+nOut+eps)), ...
        'Units','normalized','FontSize',9,'Color',[0.4 0.4 0.4], ...
        'HorizontalAlignment','right','VerticalAlignment','bottom');
end

function cmap = bluewhitered_cmap(n)
    if nargin<1, n=256; end
    half = floor(n/2);
    b2w = [linspace(0,1,half)',linspace(0,1,half)',ones(half,1)];
    w2r = [ones(n-half,1),linspace(1,0,n-half)',linspace(1,0,n-half)'];
    cmap = [b2w; w2r];
end