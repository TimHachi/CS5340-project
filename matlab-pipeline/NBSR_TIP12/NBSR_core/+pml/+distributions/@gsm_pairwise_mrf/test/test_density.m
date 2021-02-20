function test_suite = test_density
  initTestSuite;
end

function mrf = setup
  mrf = pml.distributions.gsm_pairwise_mrf;
  
  mrf.experts{1}.precision = 1 / 300;
  mrf.experts{1}.scales = exp([-6:2:6]);
  % mrf.experts{1}.weights = ones(mrf.experts{1}.nscales,1);
  mrf.experts = {mrf.experts{1}, mrf.experts{1}, mrf.experts{1}, mrf.experts{1}};
  % random expert weights
  for i = 1:mrf.nexperts, mrf.experts{i}.weights = rand(4,1); end
  
  mrf.imdims = [5 5];
end

function [f, g] = energy_plus_grad_weights(w, mrf, x)
  ntotalweights = 0;
  weight_idx = cell(1, mrf.nexperts);
  for i = 1:mrf.nexperts
    nweights = mrf.experts{i}.nscales;
    mrf.experts{i}.weights = w(1:nweights);
    w = w(nweights+1:end);
    %
    weight_idx{i} = ntotalweights+1:ntotalweights+nweights;
    ntotalweights = weight_idx{i}(end);
  end
  f = sum(mrf.energy(x));
  g = mrf.energy_grad_weights(x);
  
  % multiply g by #images
  g = g * size(x,2);
  % multiply each expert's gradient by it's #cliques
  % - case of single expert: only works if images are square
  for i = 1:mrf.nexperts
    d_filter = mrf.conv2(reshape(x(:,1), mrf.imdims), mrf.filter(i));
    g(weight_idx{i}) = g(weight_idx{i}) * numel(d_filter);
  end
  
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
  
  for i = 1:mrf.nfilters
    [frows, fcols] = size(mrf.filter_valid{i});
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