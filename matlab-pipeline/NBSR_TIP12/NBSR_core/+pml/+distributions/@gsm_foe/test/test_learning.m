function test_suite = test_learning
  initTestSuite;
end

function record = setup
  mrf = pml.distributions.gsm_foe;
  mrf.zeromean_filter = true;
  mrf.imdims = [30 30];
  
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-20:2:10]);
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales, 1);
  
  nexperts = 2;
  
  % filter_size = [3 3];
  % load(sprintf('A_%dx%d.mat', filter_size(1), filter_size(2)));
  % A = randn(size(A));
  % A = eye(prod(filter_size));
  % mrf = mrf.set_filter(A, .1*randn(prod(filter_size), nexperts), filter_size);
  
  load 'roth_3x3foe.mat'
  filter_size = [3 3];
  J_tilde = J_tilde(:,1:nexperts);
  mrf = mrf.set_filter(A, J_tilde, filter_size);
  
  mrf.experts = arrayfun(@(i) {mrf.experts{1}}, 1:nexperts);
  % mrf.J, mrf.experts{:}
  
  % load cell array X of 40 training images
  load '../../@gsm_pairwise_mrf/test/trainingImages.mat'
  
  record = struct;
  record.mrf = mrf;
  record.imgs = X;
end

function test_sm(record)
  
  report = 0;
  
  % continue existing experiment
  if true
    savefile = 'sm4_20090629-2248.mat';
    load(['./experiments/', savefile])
    npatches = -1;
    % whos, options
    
    % additional batches
    options.MaxBatches = 10;
    % convert back from str to fhandle
    %eval(['options.LearningRateFactor = ', options.LearningRateFactor, ';'])
    options.LearningRateFactor = @(batch,minibatch) 1;
    % options.LearningRate = 0.5;
    
  else
    options = struct;
    options.MaxBatches      = 10;%100;
    options.MinibatchSize   = 20;
    options.LearningRate    = 0.1;
    options.ConvergenceCheck = 0;
    options.LearningRateFactor = @(batch,minibatch) 1;
    options.RecordObjective = 2;

    % npatches = 25;
    npatches = '../../@gsm_pairwise_mrf/test/results/patches30x30x1000.mat';
    % npatches = '../../@gsm_pairwise_mrf/test/results/patches30x30x1000_gnoise-var-1.mat';
    str = 'sm4';

    c = num2cell(clock); % clock = [year month day hour minute seconds]
    savefile = sprintf('%s_%02d%02d%02d-%02d%02d.mat', str, c{1:5});
  end
  
  func = @(mrf, data, options) score_matching(mrf, data, options);
  mrf = learn(record, func, options, savefile, npatches);
  
end


function test_cd(record)
  
  options = struct;
  options.MaxBatches      = 2;%100;
  options.MinibatchSize   = 20;
  options.LearningRate    = 0.05;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1;
  
  % npatches = 25;
  % npatches = '../../@gsm_pairwise_mrf/test/results/patches30x30x1000.mat';
  npatches = '../../@gsm_pairwise_mrf/test/results/patches30x30x1000_gnoise-var-1.mat';
  niters = 1;
  str = 'cd3_noisy';
  
  c = num2cell(clock); % clock = [year month day hour minute seconds]
  savefile = sprintf('%s_%02diters_%02d%02d%02d-%02d%02d.mat', str, niters, c{1:5});
  func = @(mrf, data, options) cd(mrf, data, niters, options);
  mrf = learn(record, func, options, savefile, npatches);
end


function mrf = learn(record, func, options, savefile, npatches)
  mrf = record.mrf;
  % override matlab report function
  report = [];
  extra_elapsedtime = 0;
  done_batches = 0;
  savefile = ['./experiments/', savefile];
  
  if isstr(npatches)
    fprintf('Loading image patches from %s.\n', npatches)
    load(npatches)
    if ~exist('patches', 'var'), error('No PATCHES variable'), end
  elseif npatches == -1
    % continue experiment
    options_bck = options;
    load(savefile)%, whos, pause
    done_batches = size(report.iter_x,2) / (size(patches,2) / options.MinibatchSize);
    options = options_bck;
    % options_bck.StartBatch = done_batches;
    mrf_learned.imdims = repmat(sqrt(size(patches,1)),1,2);
    mrf = mrf_learned;
    record.report = report;
    extra_elapsedtime = learningtime;
  else
    % crop random patches from training images
    patches = pml.support.random_patches(record.imgs, mrf.imdims, npatches);
    fprintf('Cropping %d random image patches.\n', 40*npatches)
    % save patches patches, pause
  end
  
  % subtract mean from image patches
  patches = bsxfun(@minus, patches, mean(patches));
  
  if ~isempty(savefile)
    record = rmfield(record, 'imgs');
    record.patches = patches;
  end
  
  % options, done_batches, pause
  
  % learn parameters
  tic
  stepsize = 20;
  maxbatches = options.MaxBatches;
  options.MaxBatches = done_batches;
  % if isfield(options, 'StartBatch')
  %   options.MaxBatches = options.StartBatch-1;
  % else
  %   options.MaxBatches = 0;
  % end
  % donebatches = 0;
  while maxbatches > 0
    batchestodo = min(stepsize,maxbatches);
    options.StartBatch = options.MaxBatches + 1;
    options.MaxBatches = options.MaxBatches + batchestodo;
    maxbatches = maxbatches - batchestodo;
    % donebatches = donebatches + options.MaxBatches;
    
    [mrf, report] = func(mrf, patches, options);
    toc, elapsedtime = toc;
    
    record.mrf_learned = mrf;
    if isfield(record, 'report')
      % append to report
      fnames = fieldnames(record.report);
      for i = 1:length(fnames)
        cur_value = getfield(record.report, fnames{i});
        app_value = getfield(report, fnames{i});
        record.report = setfield(record.report, fnames{i}, [cur_value, app_value]);
      end
    else
      % 1st report
      record.report = report;
    end
    
    % save results
    if ~isempty(savefile)
      % record = rmfield(record, 'imgs');
      % record.patches = patches;
      record.options = options;
      if isfield(record.options, 'LearningRateFactor')
        record.options.LearningRateFactor = func2str(record.options.LearningRateFactor);
      end
      % record.options.MaxBatches = record.options.MaxBatches + donebatches;
      record.learningtime = elapsedtime + extra_elapsedtime;
      save(savefile, '-struct', 'record')
    end
    
  end
  
  
end

