function test_suite = test_statistics
  initTestSuite;
end

function record = setup
  mrf = pml.distributions.gsm_pairwise_mrf;
  
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-4.9:2.6:7]);
  
  % % uniform weights
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales,1);
  
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}};
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}, mrf.experts{1}, mrf.experts{1}};
  
  mrf.imdims = [100 100];
  
  % load cell array X of 40 training images
  load './trainingImages.mat'
  
  record = struct;
  record.mrf = mrf;
  record.imgs = X;
end

function test_scaleinvariance(record)
  mrf = record.mrf;
  
  % 5 scales with "good weights"
  mrf.experts{1}.scales = exp([-4.9:2.6:7]);
  % % CD, 1 iters
  % mrf.experts{1}.weights = [0.3908    0.3219    0.1508    0.0812    0.0553]';
  % CD, 20 iters
  mrf.experts{1}.weights = [0.4432    0.3218    0.1100    0.0664    0.0586]';
  
  load 'results/patches80x80x1000.mat'
  
  colormap jet;
  colors = colormap;
  ncolors = size(colors,1);
  colstep = 30;%floor(ncolors/3);
  
  filters{1} = [1 -1];
  sizes = {[80 80],[40 40],[20 20]};
  
  for i = 1:3
    mrf.imdims = sizes{i};
    foo = filter_responses(mrf, patches{i}, filters);
    patches_stats{i} = foo{1};
  end
  
  figure(1), clf
  subplot(2,1,1)
  
  for i = 1:3
    semilogy(patches_stats{i}.domain_grid{1}, patches_stats{i}.weights,...
             '-', 'LineWidth', 2, 'Color', colors(max(1,(i-1)*colstep),:))
    hold on
  end
  % semilogy(R, normpdf(R,0,sqrt(patches_stats{1}.covariance)), '--k', 'LineWidth', 2)
  title(sprintf('horizontal derivative of 1000 80x80 image patches for 3 spatial scales (1,2,4)'))
  legend 1 2 4
  axis([-300 300 0 1])
  drawnow
  
  % fit expert's precision to samples
  mrf.imdims = sizes{1};
  [mrf, dz] = mrf.fit_precision(patches{1});
  % dz{:}, pause
  
  npixels = prod(mrf.imdims);
  nsamples = 1000;
  niters = 15;
  
  % generate samples
  xs = 255*rand(npixels,nsamples);
  for j = 1:nsamples
    for tmp = 1:niters
      z = mrf.sample_z(xs(:,j));
      xs(:,j) = mrf.sample_x(z);
    end
    fprintf('\rSample %04d / %04d', j, nsamples)
  end
  
  samples = cell(1,3);
  samples{1} = xs;
  samples{2} = zeros(1600,nsamples);
  samples{3} = zeros(400,nsamples);
  for i = 1:nsamples
    samples{2}(:,i) = reshape(imresize(reshape(samples{1}(:,i),80,80), 1/2, 'bilinear'),1600,1);
    samples{3}(:,i) = reshape(imresize(reshape(samples{2}(:,i),40,40), 1/2, 'bilinear'),400,1);
  end
  
  % clf
  % subplot(3,1,1), imagesc(pml.support.assemble_images(samples{1}, sizes{1}, 40)), axis image
  % subplot(3,1,2), imagesc(pml.support.assemble_images(samples{2}, sizes{2}, 40)), axis image
  % subplot(3,1,3), imagesc(pml.support.assemble_images(samples{3}, sizes{3}, 40)), axis image
  % pause
  
  for i = 1:3
    mrf.imdims = sizes{i};
    foo = filter_responses(mrf, samples{i}, filters);
    sample_stats{i} = foo{1};
  end
  
  subplot(2,1,2)
  for i = 1:3
    semilogy(sample_stats{i}.domain_grid{1}, sample_stats{i}.weights,...
             '-', 'LineWidth', 2, 'Color', colors(max(1,(i-1)*colstep),:))
    hold on
  end
  % semilogy(R, normpdf(R,0,sqrt(patches_stats{1}.covariance)), '--k', 'LineWidth', 2)
  title(sprintf('horizontal derivative of 1000 80x80 samples for 3 spatial scales (1,2,4)'))
  legend 1 2 4
  axis([-300 300 0 1])
  drawnow
  
  print_pdf 'scale_invariance_80x80_124.pdf'
  
  % imagesc(pml.support.assemble_images(patches{1}, [80 80], 40)), axis image, drawnow  
  % orig = patches;
  % 
  % patches = cell(1,3);
  % patches{1} = orig;
  % patches{2} = zeros(1600,1000);
  % patches{3} = zeros(400,1000);
  % for i = 1:1000
  %   patches{2}(:,i) = reshape(imresize(reshape(patches{1}(:,i),80,80), 1/2, 'bilinear'),1600,1);
  %   patches{3}(:,i) = reshape(imresize(reshape(patches{2}(:,i),40,40), 1/2, 'bilinear'),400,1);
  % end
  % 
  % save 'results/patches80x80x1000.mat' 'patches'
  
end

function test_derivativefilter(record)
  mrf = record.mrf;
  
  experiment = 'log01';
  mrf.imdims = [30 30];
  mrf.experts{1}.precision = 1 / 7.976698905901076;
  mrf.experts{1}.scales = exp([-4.5:1/4:0]);
  mrf.experts{1}.weights = [ ...
  0.000017438429420
  0.000041338654390
  0.000118341745104
  0.000433422669543
  0.002209716499021
  0.015062812990458
  0.041430580227842
  0.046803705919537
  0.094724531317456
  0.170115259007221
  0.159225830479150
  0.123747021561127
  0.134446846217695
  0.140931573173408
  0.045798145087456
  0.014177912278928
  0.005864609638764
  0.003025298639197
  0.001825615464282 ...
  ];
  
  
  % experiment = 'sm28_scaletest_noisy';
  % load(sprintf('./results/%s/20090606-1922_sm28_scaletest_noisy_01.mat', experiment),'mrf_learned')
  % mrf.experts{1} = mrf_learned.experts{1};
  % % mrf_learned, mrf, mrf.experts{1}, pause
  
  npixels = prod(mrf.imdims);
  nsamples = 1000;
  niters = 20;
  
  % generate samples
  xs = 255*rand(npixels,nsamples);
  for j = 1:nsamples
    for tmp = 1:niters
      z = mrf.sample_z(xs(:,j));
      xs(:,j) = mrf.sample_x(z);
    end
    fprintf('\rSample %04d / %04d', j, nsamples)      
  end
  save(sprintf('%s_%dsamples', experiment, nsamples), 'xs')

  % load(sprintf('%s_%dsamples', experiment, nsamples), 'xs')
  % figure(1), clf, colormap(gray(512))
  % imagesc(pml.support.assemble_images(xs(:,1:9), mrf.imdims, 3)), axis image, drawnow
  % while true, pause, end
  
  
  filters = {[1 -1], [1 -1]'};
  nfilters = length(filters);
  colors = {'r', 'b'};
  
  % compute sample stats
  sample_stats = filter_responses(mrf, xs, filters);
  
  figure(1), clf, subplot(2,1,2)
  for i = 1:nfilters
    semilogy(sample_stats{i}.domain_grid{1}, sample_stats{i}.weights,...
             '-', 'LineWidth', 2, 'Color', colors{i})
    hold on
  end
  drawnow
  title(sprintf('x- and y-derivative of %d 30x30 samples', nsamples))
  legend x-derivative y-derivative
  axis([-300 300 0 1])
  
  
  load 'results/patches30x30x1000.mat'
  % load 'results/patches30x30x1000_gnoise-var-1.mat'
  patches_stats = filter_responses(mrf, patches, filters);
  
  subplot(2,1,1)
  for i = 1:nfilters
    semilogy(patches_stats{i}.domain_grid{1}, patches_stats{i}.weights,...
             '-', 'LineWidth', 2, 'Color', colors{i})
    hold on
  end
  % semilogy(R, normpdf(R,0,sqrt(patches_stats{1}.covariance)), '--k', 'LineWidth', 2)
  title(sprintf('x- and y-derivative of 1000 30x30 image patches'))
  legend x-derivative y-derivative
  axis([-300 300 0 1])
  drawnow
  
  print_pdf(sprintf('xy_derivative_%s.pdf', experiment))
  
  
end

function test_marginals(record)
  mrf = record.mrf;
  mrf.imdims = [30 30];
  
  % 5 scales with "good weights"
  mrf.experts{1}.scales = exp([-4.9:2.6:7]);
  % % CD, 1 iters
  % mrf.experts{1}.weights = [0.3908    0.3219    0.1508    0.0812    0.0553]';
  % % CD, 20 iters
  % mrf.experts{1}.weights = [0.4432    0.3218    0.1100    0.0664    0.0586]';
  % % SM
  mrf.experts{1}.weights = [0.1996    0.2137    0.2553    0.1943    0.1372]';
  
  % % CD, 2 experts
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}};
  % mrf.experts{1}.weights = [0.2900    0.3281    0.1913    0.1113    0.0793]';
  % mrf.experts{2}.weights = [0.5220    0.2690    0.1173    0.0526    0.0391]';
  
  %test
  %test
  experiment = 'cd23_scaletest';
  load(sprintf('./results/%s/20090529-2114_20iters_cd23_scaletest_01.mat', experiment),'mrf_learned')
  mrf.experts{1} = mrf_learned.experts{1};
  % mrf.experts{1}, pause
  
  % mrf.experts{1}.scales = exp([-10:1:10]);
  % mrf.experts{1}.weights = [0.0078    0.0081    0.0087    0.0098    0.0124    0.0199    0.0626    0.4719    0.0489    0.0405    0.0981    0.0571    0.0207    0.0685    0.0321    0.0006    0.0001    0.0000    0.0000    0.0000    0.0318];
  
  % npatches * 40 image patches
  % npatches = 25;
  % patches = pml.support.random_patches(record.imgs, mrf.imdims, npatches);
  load 'results/patches30x30x1000.mat'
  
  % fit expert's precision to samples
  [mrf, dz] = mrf.fit_precision(patches);
  % dz{:}, pause
  
  npixels = prod(mrf.imdims);
  nsamples = 1000;
  niters = 20;
  
  % generate samples
  xs = 255*rand(npixels,nsamples);
  for j = 1:nsamples
    for tmp = 1:niters
      z = mrf.sample_z(xs(:,j));
      xs(:,j) = mrf.sample_x(z);
    end
    fprintf('\rSample %04d / %04d', j, nsamples)      
  end
  
  
  for s = 2:1:8
    nfilters = 8;
    % fsize = [7 7];
    fsize = [s s];
    for i = 1:nfilters
      filters{i} = randn(fsize);
      filters{i} = filters{i} - mean(filters{i}(:));
      filters{i} = filters{i} / norm(filters{i}(:));
      % filters{i}, mean(filters{i}(:)), norm(filters{i}(:))
    end
    
    colormap jet;
    colors = colormap;
    ncolors = size(colors,1);
    colstep = floor(ncolors/nfilters);
    % R = -100:100;
    
    patches_stats = filter_responses(mrf, patches, filters);
    
    figure(1), clf
    subplot(2,1,1)
    
    for i = 1:nfilters
      semilogy(patches_stats{i}.domain_grid{1}, patches_stats{i}.weights,...
               '-', 'LineWidth', 2, 'Color', colors(max(1,(i-1)*colstep),:))
      hold on
    end
    % semilogy(R, normpdf(R,0,sqrt(patches_stats{1}.covariance)), '--k', 'LineWidth', 2)
    drawnow
    title(sprintf('%d random %dx%d filters on 1000 30x30 image patches', nfilters, fsize(1), fsize(2)))
    axis([-300 300 0 1])
    
    % compute sample stats
    sample_stats = filter_responses(mrf, xs, filters);
    
    subplot(2,1,2)
    for i = 1:nfilters
      semilogy(sample_stats{i}.domain_grid{1}, sample_stats{i}.weights,...
               '-', 'LineWidth', 2, 'Color', colors(max(1,(i-1)*colstep),:))
      hold on
    end
    drawnow
    title(sprintf('%d random %dx%d filters on 1000 30x30 samples', nfilters, fsize(1), fsize(2)))
    axis([-300 300 0 1])
    
    % title({'x- and y-derivatives of data vs. samples (1000 x 30x30)', ...
    %        'single expert with weights = (0.3908, 0.3219, 0.1508, 0.0812, 0.0553), expert''s precision fit to data',...
    %        'expert''s scales = exp([-4.9:2.6:7])'})
    % 
    % legend 'data' 'samples'
    
    print_pdf(sprintf('filter_resp_%dx%d_%s.pdf', fsize(1), fsize(2), experiment))
    % pause
    
    fprintf('\n')
  end
  
end


function dz = filter_responses(this, x, filters)
  
  nimages  = size(x, 2);
  nfilters = length(filters);
  
  fil = cell(1, nfilters);
  nelements = zeros(1, nfilters);
  switch this.conv_method
    case 'valid'
      for j = 1:nfilters
        [frows, fcols] = size(filters{j});
        % #elements of convolved image with 'valid' option
        nelements(j) = (this.imdims(1)-frows+1) * (this.imdims(2)-fcols+1);
        fil{j} = zeros(1, nelements(j) * nimages);
      end
    case 'circular'
      for j = 1:nfilters
        nelements(j) = prod(this.imdims);
        fil{j} = zeros(1, nelements(j) * nimages);
      end
    otherwise
      error('Not implemented: ''%s''.', this.conv_method);
  end
  
  for i = 1:nimages
    img = reshape(x(:, i), this.imdims);
    for j = 1:nfilters
      % apply filter
      d_filter = this.conv2(img, filters{j});
      % collect filter responses in a big vector
      fil{j}(nelements(j)*(i-1)+1:nelements(j)*i) = d_filter(:);
    end
  end
  
  dz = cell(1, nfilters);
  
  for i = 1:nfilters
    % create discrete distribution, set domain start and stride, fit to filter responses
    minval = min(fil{i});
    nweights = ceil(max(fil{i})-minval+1);
    dz{i} = pml.distributions.discrete(ones(nweights,1));
    dz{i}.domain_stride = 1;
    dz{i}.domain_start = minval*dz{i}.domain_stride;
    dz{i} = dz{i}.mle(fil{i});
  end
  
end