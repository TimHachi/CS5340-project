function test_suite = test_density
  initTestSuite;
end

function mrf = setup
  
  mrf = pml.distributions.gsm_foe;
  mrf.zeromean_filter = false;
  mrf.imdims = [8 8];
  
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-6:2:6]);
  % mrf.experts = {mrf.experts{1}, mrf.experts{1}, mrf.experts{1}, mrf.experts{1}};
  mrf.experts = {mrf.experts{1}, mrf.experts{1}};
  % random expert weights
  for i = 1:mrf.nexperts, mrf.experts{i}.weights = ones(mrf.experts{1}.nscales,1); end
  
  % filter_size = [3 3];
  % load(sprintf('A_%dx%d.mat', filter_size(1), filter_size(2)));
  % % A = randn(size(A));
  % mrf = mrf.set_filter(A, .1*randn(prod(filter_size), mrf.nexperts), filter_size);
  % load 'roth_3x3foe.mat'
  A = eye(9);
  J_tilde = randn(9,mrf.nexperts);
  filter_size = [3 3];
  mrf = mrf.set_filter(A, J_tilde, filter_size);
  % mrf = mrf.set_filter(eye(9), A'*J_tilde, filter_size);
  
end

function test_basic(mrf)
  load '/Users/uwe/Desktop/kbjasv.mat' 'img' 'J'
  % J, x = reshape(img, mrf.imdims)
  mrf.J_tilde(:) = J(:);
  % J = mrf.J
  % mrf.energy(img)
  % mrf.log_grad_x(img)
  mrf.energy_grad_weights(img)
end

function [f, g] = log_plus_grad_x(x, mrf)
  f = -mrf.energy(x);
  g = mrf.log_grad_x(x);
end

function [f, g] = energy_plus_grad_mus(m, mrf, x)
  mrf.mus = m;
  f = sum(mrf.energy(x));
  g = mrf.energy_grad_mus(x);
  
  ncliques = size(mrf.img_cliques(zeros(mrf.imdims)),2);
  nimages = size(x, 2);
  % ncliques = numel(mrf.conv2(reshape(x(:,1), mrf.imdims), mrf.filter(1)));
  f = f / (nimages * ncliques);
  % g = g * nimages * ncliques;
end

function [f, g] = energy_plus_grad_weights(w, mrf, x)
  mrf.weights = w;
  f = sum(mrf.energy(x));
  g = mrf.energy_grad_weights(x);
  
  % fprintf('%12.6f\n', f)
  % fprintf('%12.6f ', g)
  % fprintf('\n')
  
  ncliques = size(mrf.img_cliques(zeros(mrf.imdims)),2);
  nimages = size(x, 2);
  % ncliques = numel(mrf.conv2(reshape(x(:,1), mrf.imdims), mrf.filter(1)));
  f = f / (nimages * ncliques);
  % g = g * nimages * ncliques;
end

function [f, g] = energy_plus_energy_grad_J_tilde(J_tilde, mrf, x)
  mrf.J_tilde(:) = J_tilde;
  f = sum(mrf.energy(x));
  g = mrf.energy_grad_J_tilde(x);
  ncliques = size(mrf.img_cliques(zeros(mrf.imdims)),2);
  nimages = size(x, 2);
  % ncliques = numel(mrf.conv2(reshape(x(:,1), mrf.imdims), mrf.filter(1)));
  f = f / (nimages * ncliques);
  % g = g * nimages * ncliques;
end

% function [f, g] = psi_plus_grad_J_tilde(J_tilde, mrf, x)
%   mrf.J_tilde(:) = J_tilde;
%   % f = mrf.log_grad_x(x);
%   [psi, dx_psi, g_psi, g_dx_psi] = mrf.psi_grad_J_tilde(x);
%   
%   % [f, psi], pause
%   % dx_psi, pause
%   % g_dx_psi, pause
%   
%   % f = psi; g = g_psi;
%   f = dx_psi; g = g_dx_psi;
%   
%   % f, g, pause
%   % f = f(1); g = g(1,:)';
%   f = sum(sum(f)); g = sum(g,1)';
%   
%   
%   % nimages = size(x, 2);
%   % ncliques = numel(mrf.conv2(reshape(x(:,1), mrf.imdims), mrf.filter(1)));
%   % f = f / (nimages * ncliques);
%   % % g = g * nimages * ncliques;
% end

function [f, g] = sd_plus_grad(theta, mrf, x)
  [g, f] = mrf.squared_distance(theta, x);
end

function test_sd(mrf)
  x = 255 * randn(prod(mrf.imdims), 2);
  % mrf.conv_method = 'circular';
  grad_err_valid = pml.numerical.checkgrad(@sd_plus_grad, [log(mrf.weights); mrf.J_tilde(:)], 1.e-6, mrf, x)
  assertTrue(grad_err_valid < 1.e-5)
end

% function test_psi_grad_J_tilde(mrf)
%   % mrf.imdims = [5 5];
%   x = 255 * randn(prod(mrf.imdims), 1);
%   % x = ones(prod(mrf.imdims),1);
%   % x = bsxfun(@minus, x, mean(x));
%   
%   % reshape(x, mrf.imdims)
%   
%   % mrf.psi_grad_J_tilde(x)
%   
%   % grad_err_valid = pml.numerical.checkgrad(@psi_plus_grad_x, x, 1.e-6, mrf)
%   % return
%   
%   % mrf.conv_method = 'valid';
%   grad_err_valid = pml.numerical.checkgrad(@psi_plus_grad_J_tilde, mrf.J_tilde(:), 1.e-6, mrf, x)
%   assertTrue(grad_err_valid < 1.e-5)
%   
%   % mrf.conv_method = 'circular';
%   % grad_err_circular = pml.numerical.checkgrad(@psi_plus_grad_J_tilde, mrf.J_tilde(:), 1.e-6, mrf, x)
%   % assertTrue(grad_err_circular < 1.e-5)
% end

function test_energy_grad_J_tilde(mrf)
  x = 50 * randn(prod(mrf.imdims), 3);
  % x = bsxfun(@minus, x, mean(x));
  
  mrf.conv_method = 'valid';
  grad_err_valid = pml.numerical.checkgrad(@energy_plus_energy_grad_J_tilde, mrf.J_tilde(:), 1.e-6, mrf, x)
  assertTrue(grad_err_valid < 1.e-5)
  
  mrf.conv_method = 'circular';
  grad_err_circular = pml.numerical.checkgrad(@energy_plus_energy_grad_J_tilde, mrf.J_tilde(:), 1.e-6, mrf, x)
  assertTrue(grad_err_circular < 1.e-5)
end

function test_energy_grad_weights(mrf)
  % 3 random images
  x = 255 * rand(prod(mrf.imdims), 3);
  
  mrf.conv_method = 'valid';
  grad_err_valid = pml.numerical.checkgrad(@energy_plus_grad_weights, mrf.weights, 1.e-6, mrf, x)
  assertTrue(grad_err_valid < 1.e-6)
  
  mrf.conv_method = 'circular';
  grad_err_circular = pml.numerical.checkgrad(@energy_plus_grad_weights, mrf.weights, 1.e-6, mrf, x)
  assertTrue(grad_err_circular < 1.e-6)
end

function test_energy_grad_mus(mrf)
  % 3 random images
  x = 255 * rand(prod(mrf.imdims), 3);
  
  mrf.mus = 2*randn(size(mrf.mus));
  mus = mrf.mus
  
  mrf.conv_method = 'valid';
  grad_err_valid = pml.numerical.checkgrad(@energy_plus_grad_mus, mrf.mus, 1.e-6, mrf, x)
  assertTrue(grad_err_valid < 1.e-6)
  
  mrf.conv_method = 'circular';
  grad_err_circular = pml.numerical.checkgrad(@energy_plus_grad_mus, mrf.mus, 1.e-6, mrf, x)
  assertTrue(grad_err_circular < 1.e-6)
end

function test_log_grad_x(mrf)
  % random image
  x = 255 * rand(mrf.imdims);
  
  mrf.conv_method = 'valid';
  grad_err_valid = pml.numerical.checkgrad(@log_plus_grad_x, x(:), 1.e-6, mrf)
  assertTrue(grad_err_valid < 1.e-3)
  
  mrf.conv_method = 'circular';
  grad_err_circular = pml.numerical.checkgrad(@log_plus_grad_x, x(:), 1.e-6, mrf)
  assertTrue(grad_err_circular < 1.e-3)
end

function test_energy(mrf)
  % create uniform image of random intensity value
  img = pml.support.randi(255) * ones(mrf.imdims);
  npixels = prod(mrf.imdims);
  
  mrf.conv_method = 'valid';
  E0_valid = mrf.energy(img(:));
  
  mrf.conv_method = 'circular';
  E0_circular = mrf.energy(img(:));
  
  xE = - 0.5 * mrf.epsilon * img(:)'*img(:);
  E0_valid_ref = xE; E0_circular_ref = xE;
  
  mrf.conv_method = 'valid';
  
  for i = 1:mrf.nfilters
    [frows, fcols] = size(mrf.filter(i));
    % #elements of convolved image with 'valid' option
    nelements = (mrf.imdims(1)-frows+1) * (mrf.imdims(2)-fcols+1);
    % energy value from filter's expert
    ee0 = mrf.experts{min(i,mrf.nexperts)}.energy(0);
    E0_valid_ref = E0_valid_ref + nelements * ee0;
    E0_circular_ref = E0_circular_ref + npixels * ee0;
  end
  
  assertElementsAlmostEqual(E0_valid, E0_valid_ref);
  assertElementsAlmostEqual(E0_circular, E0_circular_ref);
end