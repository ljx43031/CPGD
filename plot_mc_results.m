%% plot_mc_results.m
% Monte Carlo Results Visualization for CPGD vs Baselines
% Loads results_easy/medium/hard/blackout.mat and generates all comparison figures.
%
% Figure layout:
%   Fig 1-4  : Typical trajectory comparison (one per scenario)
%   Fig 5    : Mean position RMSE bar chart (all scenarios, all methods)
%   Fig 6    : Mean velocity RMSE bar chart
%   Fig 7-10 : Maneuver-window error analysis (one per scenario)
%   Fig 11   : Blackout interval analysis (accuracy + stability)
%   Fig 12   : Blackout scenario RMSE over time (mean +/- std)
%
% Requires MATLAB R2015b or later.
% Helper functions: pick_representative_trial.m, mc_stats.m

clear; close all; clc;

% =========================================================================
% 0. Configuration
% =========================================================================
DATA_DIR   = fileparts(mfilename('fullpath'));   % same folder as this script
FIG_DIR    = fullfile(DATA_DIR, 'figures_mc');
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

SCENARIOS  = {'easy', 'medium', 'hard', 'blackout'};
SCN_LABELS = {'Easy', 'Medium', 'Hard', 'Blackout'};
THIN       = 20;    % plot one marker every THIN timesteps

% Method names (order must match fields saved by mc_compare_v8.py)
% cpgd_smooth = diffusion denoiser output only (no MMEN-KF)
% cpgd        = full pipeline: V8-Smooth + MMEN-KF
METHODS    = {'cpgd_smooth', 'cpgd', 'T_FoT_GP', 'T_FoT_StP', 'T_FoT', 'GPMT'};
MTH_LABELS = {'V8-Smooth (ours)', 'CPGD (ours)', 'T-FoT-GP', 'T-FoT-StP', 'T-FoT', 'GPMT'};
MTH_COLORS = {[0.30 0.75 0.93], [0.00 0.45 0.74], [0.85 0.33 0.10], ...
              [0.93 0.69 0.13], [0.49 0.18 0.56], [0.47 0.67 0.19]};
MTH_MARKER = {'p', 'o', 's', '^', 'd', 'v'};

% =========================================================================
% 1. Load all scenario data
% =========================================================================
D = struct();
for si = 1:numel(SCENARIOS)
    fname = fullfile(DATA_DIR, 'mc_results', ['results_' SCENARIOS{si} '.mat']);
    if ~exist(fname, 'file')
        warning('File not found: %s - skipping scenario %s', fname, SCENARIOS{si});
        D.(SCENARIOS{si}) = [];
    else
        D.(SCENARIOS{si}) = load(fname);
        fprintf('Loaded %s  (%d trials)\n', fname, size(D.(SCENARIOS{si}).states,1));
    end
end

% =========================================================================
% 2. Typical trajectory plots (Fig 1-4)
% =========================================================================
for si = 1:numel(SCENARIOS)
    scn  = SCENARIOS{si};
    scnL = SCN_LABELS{si};
    d    = D.(scn);
    if isempty(d), continue; end

    % Pick representative trial based on CPGD position error
    rep = pick_representative_trial(d.cpgd_pos_err);
    N   = size(d.states, 2);
    t   = squeeze(d.timestamps(rep,:));        % (N,)
    gt  = squeeze(d.states(rep,:,1:2));        % (N,2)  x,y

    fig = figure('Name', ['Traj_' scn], 'NumberTitle','off', ...
                 'Position', [100 100 700 600]);
    hold on; box on; grid on;

    % Ground truth - thin line + markers every THIN steps
    idx_thin = 1:THIN:N;
    plot(gt(:,1), gt(:,2), 'k-', 'LineWidth',1.2, 'DisplayName','Ground Truth');
    plot(gt(idx_thin,1), gt(idx_thin,2), 'k+', 'MarkerSize',7, 'HandleVisibility','off');

    % Each method
    for mi = 1:numel(METHODS)
        mth = METHODS{mi};
        fld = [mth '_pos'];
        if ~isfield(d, fld), continue; end
        pred = squeeze(d.(fld)(rep,:,:));      % (N,2)
        clr  = MTH_COLORS{mi};
        mrk  = MTH_MARKER{mi};
        plot(pred(:,1), pred(:,2), '-', 'Color',clr, 'LineWidth',0.8, ...
             'DisplayName', MTH_LABELS{mi});
        plot(pred(idx_thin,1), pred(idx_thin,2), mrk, 'Color',clr, ...
             'MarkerSize',5, 'MarkerFaceColor',clr, 'HandleVisibility','off');
    end

    % Blackout shading if available
    if isfield(d,'blackout_start') && ~isnan(d.blackout_start(rep))
        bs = d.blackout_start(rep);
        be = d.blackout_end(rep);
        if bs>0 && be>bs
            % Python saves 0-based indices; convert to MATLAB 1-based
            % Python slice [bs:be] = frames bs..be-1 (0-based) = bs+1..be (1-based)
            bt = (bs+1):be;
            fill([gt(bt,1); flipud(gt(bt,1))], ...
                 [gt(bt,2)-5; flipud(gt(bt,2)+5)], ...
                 [0.8 0.8 0.8], 'FaceAlpha',0.3, 'EdgeColor','none', ...
                 'DisplayName','Blackout region');
        end
    end

    legend('Location','best','FontSize',9);
    xlabel('X position (m)','FontSize',11);
    ylabel('Y position (m)','FontSize',11);
    title([scnL ' scenario - representative trial trajectory'], 'FontSize',12);
    axis equal;

    saveas(fig, fullfile(FIG_DIR, ['fig_traj_' scn '.png']));
    saveas(fig, fullfile(FIG_DIR, ['fig_traj_' scn '.pdf']));
end

% =========================================================================
% 3. Mean position / velocity RMSE bar charts (Fig 5-6)
% =========================================================================
n_scn = numel(SCENARIOS);
n_mth = numel(METHODS);

pos_rmse_mat = nan(n_scn, n_mth);
vel_rmse_mat = nan(n_scn, n_mth);

for si = 1:n_scn
    scn = SCENARIOS{si};
    d   = D.(scn);
    if isempty(d), continue; end
    for mi = 1:n_mth
        mth = METHODS{mi};
        pe_fld = [mth '_pos_err'];
        ve_fld = [mth '_vel_err'];
        if isfield(d, pe_fld)
            pos_rmse_mat(si,mi) = sqrt(nanmean(d.(pe_fld)(:)));
        end
        if isfield(d, ve_fld)
            vel_rmse_mat(si,mi) = sqrt(nanmean(d.(ve_fld)(:)));
        end
    end
end

% --- Position RMSE bar ---
fig5 = figure('Name','RMSE_pos','NumberTitle','off','Position',[150 150 750 420]);
b = bar(pos_rmse_mat, 'grouped');
for mi=1:n_mth, b(mi).FaceColor = MTH_COLORS{mi}; end
set(gca, 'XTickLabel', SCN_LABELS, 'FontSize',11);
xlabel('Scenario','FontSize',12); ylabel('Position RMSE (m)','FontSize',12);
title('Mean position RMSE over 50 MC trials','FontSize',13);
legend(MTH_LABELS,'Location','northwest','FontSize',9);
grid on; box on;
saveas(fig5, fullfile(FIG_DIR,'fig_pos_rmse_bar.png'));
saveas(fig5, fullfile(FIG_DIR,'fig_pos_rmse_bar.pdf'));

% --- Velocity RMSE bar ---
fig6 = figure('Name','RMSE_vel','NumberTitle','off','Position',[200 200 750 420]);
b = bar(vel_rmse_mat, 'grouped');
for mi=1:n_mth, b(mi).FaceColor = MTH_COLORS{mi}; end
set(gca, 'XTickLabel', SCN_LABELS, 'FontSize',11);
xlabel('Scenario','FontSize',12); ylabel('Velocity RMSE (m/s)','FontSize',12);
title('Mean velocity RMSE over 50 MC trials','FontSize',13);
legend(MTH_LABELS,'Location','northwest','FontSize',9);
grid on; box on;
saveas(fig6, fullfile(FIG_DIR,'fig_vel_rmse_bar.png'));
saveas(fig6, fullfile(FIG_DIR,'fig_vel_rmse_bar.pdf'));

% =========================================================================
% 4. Maneuver-window error analysis (Fig 7-10)
% =========================================================================
for si = 1:n_scn
    scn  = SCENARIOS{si};
    scnL = SCN_LABELS{si};
    d    = D.(scn);
    if isempty(d) || ~isfield(d,'maneuver_windows'), continue; end

    n_mc = size(d.states,1);
    WIN_HALF = 15;   % +/-15 frames around maneuver peak

    % Accumulate per-method maneuver-window errors
    mw_errors = cell(1,n_mth);
    for mi=1:n_mth, mw_errors{mi} = []; end

    for tr = 1:n_mc
        % d.maneuver_windows is a cell array; each cell is a (K,3) int matrix
        % columns: [start_idx, peak_idx, end_idx] — Python 0-based indices
        mw = d.maneuver_windows{tr};   % (K, 3) matrix
        for wi = 1:size(mw, 1)
            pk = mw(wi, 2) + 1;   % peak_idx: convert 0-based -> 1-based
            t1 = max(1, pk - WIN_HALF);
            t2 = min(size(d.states,2), pk + WIN_HALF);
            seg = t1:t2;
            for mi = 1:n_mth
                pe_fld = [METHODS{mi} '_pos_err'];
                if ~isfield(d, pe_fld), continue; end
                errs = d.(pe_fld)(tr, seg);
                mw_errors{mi} = [mw_errors{mi}, errs(~isnan(errs))];
            end
        end
    end

    % Build matrix for boxplot (pad with NaN to equal length)
    max_len = max(cellfun(@numel, mw_errors));
    if max_len == 0, continue; end
    bx_mat = nan(max_len, n_mth);
    for mi=1:n_mth
        v = mw_errors{mi};
        bx_mat(1:numel(v), mi) = v(:);
    end

    fig = figure('Name',['Maneuver_' scn],'NumberTitle','off','Position',[250 250 650 420]);
    bp  = boxplot(bx_mat, 'Labels',MTH_LABELS, 'Widths',0.5, ...
                  'OutlierSize',3, 'Symbol','.');
    % Colour each box
    h = findobj(gca,'Tag','Box');
    for mi=1:min(n_mth, numel(h))
        patch(get(h(mi),'XData'), get(h(mi),'YData'), ...
              MTH_COLORS{n_mth+1-mi}, 'FaceAlpha',0.4);
    end
    ylabel('Position error in maneuver window (m)','FontSize',11);
    title([scnL ' - error distribution at sharp maneuver transitions'], 'FontSize',12);
    grid on; box on;
    saveas(fig, fullfile(FIG_DIR, ['fig_maneuver_' scn '.png']));
    saveas(fig, fullfile(FIG_DIR, ['fig_maneuver_' scn '.pdf']));
end

% =========================================================================
% 5. Blackout interval analysis (Fig 11-12)
% =========================================================================
d_bo = D.('blackout');
if ~isempty(d_bo) && isfield(d_bo,'blackout_start')

    n_mc = size(d_bo.states,1);
    N    = size(d_bo.states,2);

    REL_PRE  = 30;   % frames before blackout start
    REL_POST = 80;   % frames after blackout end (blackout_len=50)
    rel_len  = REL_PRE + 50 + REL_POST;

    acc_err  = nan(n_mth, n_mc, rel_len);

    for tr = 1:n_mc
        bs = d_bo.blackout_start(tr);
        be = d_bo.blackout_end(tr);
        if isnan(bs) || bs<=0, continue; end
        % Python saves 0-based indices; convert to MATLAB 1-based
        % Python slice [bs:be] = frames bs..be-1 (0-based) = bs+1..be (1-based)
        rel_idx = (bs + 1 - REL_PRE):(be + REL_POST);
        for mi = 1:n_mth
            pe_fld = [METHODS{mi} '_pos_err'];
            if ~isfield(d_bo, pe_fld), continue; end
            for ri = 1:rel_len
                ai = rel_idx(ri);
                if ai>=1 && ai<=N
                    acc_err(mi, tr, ri) = d_bo.(pe_fld)(tr, ai);
                end
            end
        end
    end

    rel_t = (-REL_PRE):(50 + REL_POST - 1);   % relative frame axis

    fig11 = figure('Name','Blackout_analysis','NumberTitle','off', ...
                   'Position',[300 100 850 500]);
    hold on; box on; grid on;

    for mi = 1:n_mth
        mu = squeeze(nanmean(acc_err(mi,:,:), 2));   % (rel_len,)
        sg = squeeze(nanstd( acc_err(mi,:,:), 0, 2));
        clr = MTH_COLORS{mi};
        fill([rel_t, fliplr(rel_t)], [mu'+sg'; flipud(mu'-sg')], ...
             clr, 'FaceAlpha',0.12, 'EdgeColor','none', 'HandleVisibility','off');
        plot(rel_t, mu, '-', 'Color',clr, 'LineWidth',1.6, 'DisplayName',MTH_LABELS{mi});
    end

    % Shade blackout region (replaces xregion, not available before R2021b)
    yl = ylim;
    patch([0 50 50 0], [yl(1) yl(1) yl(2) yl(2)], ...
          [0.7 0.7 0.7], 'FaceAlpha',0.25, 'EdgeColor','none', ...
          'DisplayName','Blackout');

    % Vertical lines at blackout boundaries (replaces xline, not available before R2019b)
    yl2 = ylim;
    plot([0 0],  yl2, '--k', 'LineWidth',1, 'HandleVisibility','off');
    plot([50 50], yl2, '--k', 'LineWidth',1, 'HandleVisibility','off');

    legend('Location','best','FontSize',9);
    xlabel('Frame relative to blackout start','FontSize',11);
    ylabel('Mean position error (m)','FontSize',11);
    title('Tracking error during and around blackout interval (50 MC trials)','FontSize',12);
    xlim([rel_t(1) rel_t(end)]);

    saveas(fig11, fullfile(FIG_DIR,'fig_blackout_analysis.png'));
    saveas(fig11, fullfile(FIG_DIR,'fig_blackout_analysis.pdf'));

    % --- Stability: std of position error inside blackout vs. outside ---
    fig12 = figure('Name','Blackout_stability','NumberTitle','off', ...
                   'Position',[350 150 650 420]);
    bo_inside  = nan(n_mth,1);
    bo_outside = nan(n_mth,1);
    for mi = 1:n_mth
        inside_mask  = rel_t>=0 & rel_t<50;
        outside_mask = rel_t<0;
        bo_inside(mi)  = nanmean(nanstd(squeeze(acc_err(mi,:,inside_mask)), 0, 1));
        bo_outside(mi) = nanmean(nanstd(squeeze(acc_err(mi,:,outside_mask)),0, 1));
    end
    b = bar([bo_outside, bo_inside], 'grouped');
    b(1).FaceColor = [0.4 0.7 1.0]; b(1).DisplayName = 'Pre-blackout (stable)';
    b(2).FaceColor = [1.0 0.4 0.3]; b(2).DisplayName = 'During blackout';
    set(gca,'XTickLabel',MTH_LABELS,'XTickLabelRotation',15,'FontSize',10);
    ylabel('Mean std of position error (m)','FontSize',11);
    title('Prediction stability: pre-blackout vs. during blackout','FontSize',12);
    legend({'Pre-blackout','During blackout'},'Location','northwest','FontSize',9);
    grid on; box on;
    saveas(fig12, fullfile(FIG_DIR,'fig_blackout_stability.png'));
    saveas(fig12, fullfile(FIG_DIR,'fig_blackout_stability.pdf'));
end

% =========================================================================
% 6. Per-timestep error ribbon (mean +/- std) for each scenario (Fig 13-16)
% =========================================================================
for si = 1:n_scn
    scn  = SCENARIOS{si};
    scnL = SCN_LABELS{si};
    d    = D.(scn);
    if isempty(d), continue; end
    N = size(d.states,2);
    t = 1:N;

    fig = figure('Name',['TimeSeries_' scn],'NumberTitle','off', ...
                 'Position',[100+si*40 300 800 420]);
    hold on; box on; grid on;

    for mi = 1:n_mth
        pe_fld = [METHODS{mi} '_pos_err'];
        if ~isfield(d, pe_fld), continue; end
        [mu, sg] = mc_stats(d.(pe_fld));       % (1,N)
        clr = MTH_COLORS{mi};
        fill([t, fliplr(t)], [mu+sg, fliplr(mu-sg)], ...
             clr, 'FaceAlpha',0.12, 'EdgeColor','none','HandleVisibility','off');
        plot(t, mu, '-', 'Color',clr, 'LineWidth',1.4, 'DisplayName',MTH_LABELS{mi});
    end

    legend('Location','best','FontSize',9);
    xlabel('Frame index','FontSize',11);
    ylabel('Position error (m)','FontSize',11);
    title([scnL ' - mean position error over time (50 MC trials)'],'FontSize',12);
    xlim([1 N]);

    saveas(fig, fullfile(FIG_DIR, ['fig_timeseries_' scn '.png']));
    saveas(fig, fullfile(FIG_DIR, ['fig_timeseries_' scn '.pdf']));
end

% =========================================================================
% 7. Summary statistics table (printed to console + saved as CSV)
% =========================================================================
fprintf('\n========== Summary: Mean Position RMSE (m) ==========\n');
fprintf('%-20s', 'Method');
for si=1:n_scn, fprintf('%12s', SCN_LABELS{si}); end
fprintf('\n');
for mi=1:n_mth
    fprintf('%-20s', MTH_LABELS{mi});
    for si=1:n_scn
        if ~isnan(pos_rmse_mat(si,mi))
            fprintf('%12.3f', pos_rmse_mat(si,mi));
        else
            fprintf('%12s', 'N/A');
        end
    end
    fprintf('\n');
end

fprintf('\n========== Summary: Mean Velocity RMSE (m/s) ==========\n');
fprintf('%-20s', 'Method');
for si=1:n_scn, fprintf('%12s', SCN_LABELS{si}); end
fprintf('\n');
for mi=1:n_mth
    fprintf('%-20s', MTH_LABELS{mi});
    for si=1:n_scn
        if ~isnan(vel_rmse_mat(si,mi))
            fprintf('%12.3f', vel_rmse_mat(si,mi));
        else
            fprintf('%12s', 'N/A');
        end
    end
    fprintf('\n');
end

% Save CSV
fid = fopen(fullfile(FIG_DIR,'summary_pos_rmse.csv'),'w');
fprintf(fid,'Method,%s\n', strjoin(SCN_LABELS,','));
for mi=1:n_mth
    fprintf(fid,'%s', MTH_LABELS{mi});
    for si=1:n_scn
        if ~isnan(pos_rmse_mat(si,mi))
            fprintf(fid,',%.4f', pos_rmse_mat(si,mi));
        else
            fprintf(fid,',NaN');
        end
    end
    fprintf(fid,'\n');
end
fclose(fid);

fprintf('\nAll figures saved to: %s\n', FIG_DIR);
fprintf('Done.\n');
