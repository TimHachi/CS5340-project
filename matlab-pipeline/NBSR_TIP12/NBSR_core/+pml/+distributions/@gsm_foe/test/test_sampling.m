function test_suite = test_sampling
  initTestSuite;
end

function mrf = setup
  % FIXME valid convolution seems to be broken with gsm_foe!
  % pmrf = pml.distributions.gsm_pairwise_mrf;
  mrf = pml.distributions.gsm_foe;
  mrf.zeromean_filter = true;
  mrf.imdims = [50 50];
  mrf.imdims = [30 30];
  
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-5:1:5]);
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales, 1);
  
  nexperts = 4;
  
  % pmrf.imdims = mrf.imdims;
  % pmrf.experts = mrf.experts;
  % % mrf, pmrf, pause
  % full(mrf.filter_matrices{1}), full(pmrf.filter_matrices{1}), pause
  
  % FIXME 'valid' with zero-padded derivative-filter or uneven filter size doesn't work. Is this ok?
  % mrf.J = [[1 0 -1 0]', ...
  %          [1 -1 0 0]', ...
  %          [1 0 0 -1]'*sqrt(2), ...
  %          [0 1 -1 0]'*sqrt(2)];
  
  filter_size = [3 3];
  A = eye(prod(filter_size));
  f = randn(prod(filter_size), nexperts);
  % f = bsxfun(@minus, f, mean(f));
  % f = bsxfun(@rdivide, f, arrayfun(@(i) norm(f(:,i)), 1:nexperts));
  mrf = mrf.set_filter(A, f, filter_size);
  
  % uniform weights
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales,1);
  mrf.experts = arrayfun(@(i) {mrf.experts{1}}, 1:nexperts);
  
  % print filter
  mrf.filter{:}
  
end

function test_main(mrf)
  
  expert = mrf.experts{1};
  vars = (1 ./ (mrf.experts{1}.scales * mrf.experts{1}.precision));
  for i = 1:expert.nscales
    fprintf('z = %d  =>  N(0, %6.1f)\n', i, vars(i))
  end
  
  img = zeros(mrf.imdims);
  s = mrf.imdims(1);
  img(6:s-5,6:s-5) = 255;
  img(11:s-10,11:s-10) = 0;
  for i = 1:6
    img(11-i:11-i+2,19+i) = 0;
    img(20+i-2:20+i,19+i) = 0;
  end
  
  N = mrf.nfilters + 2;
  figure(1), clf
  % figure
  colormap(gray(256))
  
  %% VALID
  mrf.conv_method = 'valid';
  z = mrf.sample_z(img(:));
  x_1 = mrf.sample_x(z);
  subplot1(2, N)
  subplot1(1), imagesc(img), title 'x^{(0)}', ylabel 'valid', axis image, colorbar
  for i = 1:mrf.nfilters
    [frows, fcols] = size(mrf.filter(i));
    % size of convolved image with 'valid' option
    csize = [mrf.imdims(1)-frows+1, mrf.imdims(2)-fcols+1];
    % convert sampled scales back to to scale indices (for the sake of visualization)
    for j = 1:expert.nscales, z{i}(z{i} == expert.scales(j)) = j; end
    subplot1(i+1), imagesc(reshape(z{i}, csize))
    title(sprintf('z^{(1)} - filter %d', i)), axis image, colorbar
  end
  subplot1(N), imagesc(reshape(x_1, mrf.imdims)), title 'x^{(1)}', axis image, colorbar
  
  % ----------------
  
  %% CIRCULAR
  mrf.conv_method = 'circular';
  z = mrf.sample_z(img(:));
  x_1 = mrf.sample_x(z);
  
  subplot1(N+1), imagesc(img), title 'x^{(0)}', ylabel 'circular', axis image, colorbar
  for i = 1:mrf.nfilters
    % convert sampled scales back to to scale indices (for the sake of visualization)
    for j = 1:expert.nscales, z{i}(z{i} == expert.scales(j)) = j; end
    subplot1(N+i+1), imagesc(reshape(z{i}, mrf.imdims))
    title(sprintf('z^{(1)} - filter %d', i)), axis image, colorbar
  end
  subplot1(N+N), imagesc(reshape(x_1, mrf.imdims)), title 'x^{(1)}', axis image, colorbar
  
  % check results visually, close figure
  % pause, close
end


function test_filter_matrices(mrf)
  % random image
  img = 255*rand(mrf.imdims);
  
  method = {'valid', 'circular'};
  for iter = 1:length(method)
    % set convolution method
    mrf.conv_method = method{iter};
    % filter matrices
    F = mrf.filter_matrices;
    for i = 1:length(F)
      % convolution by matrix multiplication
      Fconv = F{i} * img(:);
      % convolution by calling mrf's convolution function
      reference = mrf.conv2(img, mrf.filter(i));
      % check if results are equal
      assertVectorsAlmostEqual(Fconv, reference(:))
    end
  end
end


function test_sanity(mrf)
  nsamples = 5;
  method = {'valid', 'circular'};
  for iter = 1:length(method)
    % set convolution method
    mrf.conv_method = method{iter};
    
    % random images
    r = 255*rand(prod(mrf.imdims), nsamples);
    
    % samples from the MRF
    x = r;
    for i = 1:nsamples
      for tmp = 1:10
        z = mrf.sample_z(x(:, i));
        x(:, i) = mrf.sample_x(z);
      end
    end
    
    % sample's energy should be lower than energy of a random image
    energy_samples = mrf.energy(x)
    energy_random = mrf.energy(r)
    assertTrue(all(energy_samples < energy_random))
  end
end


% function test_visual(mrf)
%   % mrf.conv_method = 'valid';
%   mrf.conv_method = 'circular';
%   mrf.imdims = [100 100];
%   nsamples = 3;
%   niters = 7;
%   
%   % samples from the MRF
%   x = 255*rand(prod(mrf.imdims), nsamples);
%   for i = 1:nsamples
%     for tmp = 1:niters
%       z = mrf.sample_z(x(:, i));
%       x(:, i) = mrf.sample_x(z);
%     end
%   end
%   figure(1), clf, colormap(gray(256))
%   for i = 1:nsamples
%     subplot(1,nsamples,i), imagesc(reshape(x(:,i), mrf.imdims)), axis image, colorbar
%   end
% end