function test_suite = test_learning
  initTestSuite;
end

function record = setup
  mrf = pml.distributions.gsm_pairwise_mrf;
  
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-5:5]);
  
  % % uniform weights
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales,1);
  % mrf.experts{1}.weights = [.7 .1 .1 .1]';
  % mrf.experts{1}.weights = [.1 0 0 .9]';
  
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}};
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}, mrf.experts{1}, mrf.experts{1}};
  
  mrf.imdims = [30 30];
  
  % load cell array X of 40 training images
  load './trainingImages_double.mat'
  
  record = struct;
  record.mrf = mrf;
  record.imgs = X;
end

function plot_results(mrf, dz)
  % show fitted experts
  for j = 1:mrf.nexperts
    if j == 1, figure(2), clf; end
    subplot(2,2,j),
    plotyy(dz{j}.domain_grid{1}, dz{j}.weights, dz{j}.domain_grid{1}, dz{j}.weights, @semilogy, @plot)
    % dz{j}.semilogy
    title(sprintf('Expert %d, Var = %.2f, Weights = %s', j, 1/mrf.experts{j}.precision, num2str(mrf.experts{j}.weights')))
    hold on
    vals = mrf.experts{j}.eval(dz{j}.domain_grid{1});
    [ax,h1,h2] = plotyy(dz{j}.domain_grid{1}, vals, dz{j}.domain_grid{1}, vals, @semilogy, @plot);
    set(h1,'color','r'), set(h2,'color','r')
    % semilogy(dz{j}.domain_grid{1}, new_experts{j}.eval(dz{j}.domain_grid{1}), '-r');
  end
  drawnow
end

function [mrf, dz] = learn(record, func, options, doplot, savefile, npatches)
  mrf = record.mrf;
  
  if npatches > 0
    % crop random patches from training images
    patches = pml.support.random_patches(record.imgs, mrf.imdims, npatches);
    fprintf('Cropping %d random image patches.\n', 40*npatches)
    % save patches patches, pause
  else
    % CHANGED: use fixed image patches to compare learning algorithms
    load './patches.mat'
    fprintf('Loading random image patches from file.\n')
  end
  
  if doplot
    figure(1), clf, colormap(gray(256))
    imagesc(pml.support.assemble_images(patches, mrf.imdims, 40)), axis image, drawnow
  end
  
  % CHANGED: need to call this to set empirical variance before learning
  [mrf, dz] = mrf.fit_precision(patches);
  record.dz = dz;
  record.mrf = mrf;
  
  % FIXME empirical variance of MLE is unlike that of the other learning algorithms!
  % mrf = mrf.mle(patches);
  % set uniform weights again
  % for i = 1:mrf.nexperts, mrf.experts{i}.weights = ones(4,1); end
  
  if strcmp(mrf.conv_method, 'circular')
    % log-likelihood bounds before learning expert's weights
    pre_llh_bounds = mrf.log_lh_bounds(patches);
  end
  
  % weights before learning
  pre_weights = zeros(mrf.nexperts, mrf.experts{1}.nscales);
  for i = 1:mrf.nexperts, pre_weights(i,:) = mrf.experts{i}.weights'; end
  
  % learn weights
  tic
  [mrf, report] = func(mrf, patches, options);
  toc, elapsedtime = toc;
  record.mrf_learned = mrf;
  record.report = report;
  
  if doplot
    plot_results(mrf, dz);
  end
  
  % show fitted weights and difference
  post_weights = zeros(mrf.nexperts, mrf.experts{1}.nscales);
  for i = 1:mrf.nexperts, post_weights(i,:) = mrf.experts{i}.weights'; end
  pre_weights, post_weights, diff_weights = post_weights - pre_weights
  
  if strcmp(mrf.conv_method, 'circular')
    % log-likelihood bounds after learning expert's weights
    post_llh_bounds = mrf.log_lh_bounds(patches);
    
    diff_llh_bounds = post_llh_bounds - pre_llh_bounds
    % all bounds should have improved
    % assertTrue(all(post_llh_bounds(:) > pre_llh_bounds(:)))
  end
  
  % save results
  if ~isempty(savefile)
    record = rmfield(record, 'imgs');
    record.patches = patches;
    if isfield(options, 'LearningRateFactor')
      options.LearningRateFactor = func2str(options.LearningRateFactor);
    end
    if isfield(options, 'BoundTransformer')
      options.BoundTransformer = func2str(options.BoundTransformer);
    end
    record.options = options;
    record.learningtime = elapsedtime;
    save(savefile, '-struct', 'record')
  end
  
end

function test_cd(record)
  
  % add another smaller scale
  record.mrf.experts{1}.scales = exp([-4.9:2.6:7]);
  record.mrf.experts{1}.weights = ones(5,1);
  
  record.mrf.imdims = [30 30];
  record.mrf.experts = {record.mrf.experts{1}};
  % record.mrf.experts = {record.mrf.experts{1}, record.mrf.experts{1}};
  % record.mrf.conv_method = 'circular';
  
  options = struct;
  options.MaxBatches      = 0;%100;
  options.MinibatchSize   = 20;
  options.LearningRate    = 0.0001;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1 / (max(1, batch-90))^2;
  
  npatches = 25;    % #patches from each of the 40 training images (=> #training images = npatches * 40)
  doplot = false;   % plot results
  % savefile = [];    % save results to given filename
  % niters = 1; % #iters of the markov chain
  % func = @(mrf, data, options) cd(mrf, data, niters, options);
  
  niters(1) = 1;
  niters(2) = 20;
  
  nexperiments = length(niters);
  nruns = 3;
  weights = zeros(nruns, record.mrf.experts{1}.nscales);
  str = 'cd1';
  
  for j = 1:nexperiments
    
    filelist = {};
    for i = 1:nruns
      c = num2cell(clock); % clock = [year month day hour minute seconds]
      savefile = sprintf('%02d%02d%02d-%02d%02d_%02diters_%s_%02d.mat', c{1:5}, niters(j), str, i);
      filelist{i} = [savefile ' '];
      func = @(mrf, data, options) cd(mrf, data, niters(j), options);
      mrf = learn(record, func, options, doplot, savefile, npatches);
      weights(i,:) = mrf.experts{1}.weights';
      fprintf('\n==================================================\n\n')
    end

    fprintf('\nWEIGHTS:\n')
    disp(weights)
    % save 'learned_weights.mat' 'weights';

    % fileliststr = cat(2, filelist{:}, 'learned_weights.mat');
    fileliststr = cat(2, filelist{:});
    zipstr = sprintf('zip %s%02d.zip %s', str, j, fileliststr);
    rmstr = sprintf('rm %s', fileliststr);

    [status, result] = system(zipstr);
    if status == 0
      [status, result] = system(rmstr);
    else
      warning('%s', result), pause
    end
    
  end
  
end

function test_score_matching(record)
  
  record.mrf.imdims = [30 30];
  record.mrf.experts = {record.mrf.experts{1}};
  % record.mrf.experts = {record.mrf.experts{1}, record.mrf.experts{1}};
  % record.mrf.conv_method = 'circular';
  
  % base options
  options = struct;
  options.MaxBatches      = 200;
  options.MinibatchSize   = 200;
  options.LearningRate    = 0.00001;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1 / (max(1, batch-90))^2;
  
  % options for all experiments
  options(end+1) = options(end); options(end).LearningRate = 0.0001;
  options(end+1) = options(end); options(end).LearningRate = 0.001;
  
  % [options.MinibatchSize; options.LearningRate]'
  % length(options)
  % pause, pause
  
  npatches = 25;    % #patches from each of the 40 training images (=> #training images = npatches * 40)
  doplot = false;   % plot results
  % savefile = [];    % save results to given filename
  func = @(mrf, data, options) score_matching(mrf, data, options);
  
  nexperiments = length(options);
  nruns = 5;
  weights = zeros(nruns, 4);
  
  for j = 1:nexperiments
    
    filelist = {};
    for i = 1:nruns
      c = num2cell(clock); % clock = [year month day hour minute seconds]
      savefile = sprintf('%02d%02d%02d-%02d%02d_sm_%02d.mat', c{1:5}, i);
      filelist{i} = [savefile ' '];
      mrf = learn(record, func, options(j), doplot, savefile, npatches);
      weights(i,:) = mrf.experts{1}.weights';
      fprintf('\n==================================================\n\n')
    end

    fprintf('\nWEIGHTS:\n')
    disp(weights)
    save 'learned_weights.mat' 'weights';

    fileliststr = cat(2, filelist{:}, 'learned_weights.mat');
    zipstr = sprintf('zip sm2%02d.zip %s', j, fileliststr);
    rmstr = sprintf('rm %s', fileliststr);

    [status, result] = system(zipstr);
    if status == 0
      [status, result] = system(rmstr);
    else
      warning('%s', result), pause
    end
    
  end
  
end

function test_mle_mcmc(record)
  
  record.mrf.imdims = [30 30];
  record.mrf.experts = {record.mrf.experts{1}};
  % record.mrf.experts = {record.mrf.experts{1}, record.mrf.experts{1}};
  % record.mrf.conv_method = 'circular';
  
  % base options
  options = struct;
  options.MaxBatches      = 0;%100;
  options.MinibatchSize   = 20;
  options.LearningRate    = 0.0001;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1 / (max(1, batch-90))^2;
  
  % options for all experiments
  options(end+1) = options(end); options(end).LearningRate = 0.001;
  
  % [options.MinibatchSize; options.LearningRate]'
  % length(options)
  % pause, pause
  
  npatches = 1;    % #patches from each of the 40 training images (=> #training images = npatches * 40)
  doplot = false;   % plot results
  % savefile = [];    % save results to given filename
  niters = 1; % #iters of the markov chain
  func = @(mrf, data, options) mle_mcmc(mrf, data, -1, niters, options);
  
  
  nexperiments = length(options);
  nruns = 5;
  weights = zeros(nruns, 4);
  
  for j = 1:nexperiments
    
    filelist = {};
    for i = 1:nruns
      c = num2cell(clock); % clock = [year month day hour minute seconds]
      savefile = sprintf('%02d%02d%02d-%02d%02d_%02diters_ml_%02d.mat', c{1:5}, niters, i);
      filelist{i} = [savefile ' '];
      mrf = learn(record, func, options(j), doplot, savefile, npatches);
      weights(i,:) = mrf.experts{1}.weights';
      fprintf('\n==================================================\n\n')
    end

    fprintf('\nWEIGHTS:\n')
    disp(weights)
    % save 'learned_weights.mat' 'weights';

    % fileliststr = cat(2, filelist{:}, 'learned_weights.mat');
    fileliststr = cat(2, filelist{:});
    zipstr = sprintf('zip ml%02d.zip %s', j, fileliststr);
    rmstr = sprintf('rm %s', fileliststr);

    [status, result] = system(zipstr);
    if status == 0
      [status, result] = system(rmstr);
    else
      warning('%s', result), pause
    end
    
  end
end

function test_foo(record)
  record.mrf.imdims = [10 10];
  % record.mrf.experts = {record.mrf.experts{1}};
  % record.mrf.weights = rand(size(record.mrf.weights));
  
  % record.mrf = record.mrf.set_filter(eye(4), randn(4,2), [2 2]);
  % record.mrf.conditional_sampling = true;
  % record.mrf
  % return
  
  
  % base options
  options = struct;
  options.MaxBatches      = 10;
  options.MinibatchSize   = 10;
  options.LearningRate    = 0.1;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1;
  % options.RecordObjective = 2;
  
  npatches = 1;    % #patches from each of the 40 training images (=> #training images = npatches * 40)
  doplot = false;   % plot results
  func = @(mrf, data, options) score_matching(mrf, data(:,1:3), options);
  % func = @(mrf, data, options) cd(mrf, data, 1, options);
  
  mrf = learn(record, func, options, doplot, 'foo.mat', npatches);

end


function test_mle(record)
  
  record.mrf.imdims = [30 30];
  record.mrf.experts = {record.mrf.experts{1}};
  % record.mrf.experts = {record.mrf.experts{1}, record.mrf.experts{1}};
  % record.mrf.conv_method = 'circular';
  
  npatches = 25;    % #patches from each of the 40 training images (=> #training images = npatches * 40)
  doplot = false;   % plot results
  % savefile = [];    % save results to given filename
  func = @(mrf, data, options) mle(mrf, data);
  
  nexperiments = 1;
  nruns = 5;
  weights = zeros(nruns, 4);
  
  for j = 1:nexperiments
    
    filelist = {};
    for i = 1:nruns
      c = num2cell(clock); % clock = [year month day hour minute seconds]
      savefile = sprintf('%02d%02d%02d-%02d%02d_mlp_%02d.mat', c{1:5}, i);
      filelist{i} = [savefile ' '];
      mrf = learn(record, func, struct, doplot, savefile, npatches);
      weights(i,:) = mrf.experts{1}.weights';
      fprintf('\n==================================================\n\n')
    end

    fprintf('\nWEIGHTS:\n')
    disp(weights)

    fileliststr = cat(2, filelist{:});
    zipstr = sprintf('zip mlp%02d.zip %s', j, fileliststr);
    rmstr = sprintf('rm %s', fileliststr);

    [status, result] = system(zipstr);
    if status == 0
      [status, result] = system(rmstr);
    else
      warning('%s', result), pause
    end
    
  end
  
end

% function test_mle(record)
%   mrf = record.mrf;
%   
%   nimages = 40; % up to 40
%   % crop random patch from random training images
%   patches = pml.support.random_patches(record.imgs(randsample(length(record.imgs), nimages)), mrf.imdims);
%   
%   figure(1), clf, colormap(gray(256)),
%   imagesc(pml.support.assemble_images(patches, mrf.imdims, 20)), axis image, drawnow
%   
%   mrf.conv_method = 'circular';
%   
%   % FIXME: doesn't make sense since the expert's variance is different
%   % log-likelihood bounds before learning expert's weights
%   pre_llh_bounds = mrf.log_lh_bounds(patches);
%   
%   % learn weights
%   [mrf, dz] = mrf.mle(patches);
%   
%   % show fitted experts
%   for j = 1:mrf.nexperts
%     if j == 1, figure(2), clf; end
%     subplot(2,2,j),
%     plotyy(dz{j}.domain_grid{1}, dz{j}.weights, dz{j}.domain_grid{1}, dz{j}.weights, @semilogy, @plot)
%     % dz{j}.semilogy
%     title(['Expert ' num2str(j)]), hold on
%     vals = mrf.experts{j}.eval(dz{j}.domain_grid{1});
%     [ax,h1,h2] = plotyy(dz{j}.domain_grid{1}, vals, dz{j}.domain_grid{1}, vals, @semilogy, @plot);
%     set(h1,'color','r'), set(h2,'color','r')
%     % semilogy(dz{j}.domain_grid{1}, new_experts{j}.eval(dz{j}.domain_grid{1}), '-r');
%   end
%   drawnow
%   
%   % show fitted weights
%   for expert = mrf.experts, expert{1}.weights', end
%     
%   % log-likelihood bounds after learning expert's weights
%   post_llh_bounds = mrf.log_lh_bounds(patches);
%   
%   % all bounds should have improved
%   assertTrue(all(post_llh_bounds(:) > pre_llh_bounds(:)))
%   
%   % s = mrf.sample(3, 5);
%   % figure(1), clf, colormap(gray(256))
%   % for i = 1:3
%   %   subplot(1,3,i), imagesc(reshape(s(:,i), mrf.imdims)), axis image
%   %   colorbar
%   % end
%   
% end
