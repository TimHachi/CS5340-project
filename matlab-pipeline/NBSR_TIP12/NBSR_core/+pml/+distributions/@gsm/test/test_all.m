function [] = test_all()
  function [] = test(str)
    disp(str);
    eval(str);
  end

  function [f, g] = log_plus_grad_x(x, d)
    f = d.log(x);
    g = d.log_grad_x(x);
  end
  
  function [f, g] = log_plus_grad_precision(P, d, x)
    if (iscell(d.precision))
      sz = d.ndims^2;
      for i = 1:d.nscales
        d.precision{i} = reshape(P(1+(i-1)*sz:i*sz), ...
                                 size(d.precision{i}));
      end
    else
      d.precision = reshape(P, size(d.precision));
    end
    f = sum(d.log(x));
    g = d.log_grad_precision(x);
    if (iscell(d.precision))
      out = [];
      for i = 1:d.nscales
        out = [out; g{i}(:)];
      end
      g = out;
    else
      g = g(:);
    end
  end
  
  function [f, g] = log_plus_grad_weights(w, d, x)
    d.weights = w;
    f = sum(d.log(x));
    g = d.log_grad_weights(x);
    g = g(:);
  end
  
  function [] = test_probs(d, x)
    p = d.eval(x)
    u = d.unnorm(x)
    l = d.log(x)
    e = d.energy(x)
    
    max(abs((p ./ sum(p)) - (u ./ sum(u))))
    max(abs(log(p) - l))
    max(abs(log(u) + e))
    
    err = [];
    for i = 1:size(x, 2)
      err = [err, pml.numerical.checkgrad(@log_plus_grad_x, x(:, i), 1e-6, d)];
    end
    err
    
    disp('precision...')
    if (iscell(d.precision))
      P = [];
      for i = 1:d.nscales
        P = [P; d.precision{i}(:)];
      end
      pml.numerical.checkgrad(@log_plus_grad_precision, P, 1e-6, ...
                              d, x)
    else
      pml.numerical.checkgrad(@log_plus_grad_precision, d.precision(:), 1e-6, ...
                              d, x)
    end
    
    disp('weights...')
    pml.numerical.checkgrad(@log_plus_grad_weights, d.weights(:), 1e-6, ...
                            d, x)
  end

  %% 1D tests
  d = pml.distributions.gsm(1, 4);
  d.mean = 1;
  d.precision = 1;
  d.scales = logspace(0.1, 2, 4);

  % test('d.ndims')
  % test('d.mean')
  % 
  % test('d.eval(0:0.1:0.9)')
  % test('d.unnorm(0:0.1:0.9)')
  % test('d.log(0:0.1:0.9)')
  % test('d.energy(0:0.1:0.9)')
  % test('d.log_grad_x(0:0.1:0.9)')
  % test('d.sample(10)')
  % pause
  
  test_probs(d, 2:0.1:4);
  pause

  %% 1D tests + separate precision
  d.precision = {0.5, 0.75, 1, 1.25};
  test_probs(d, 2:0.1:4);
  pause


  
  %% 2D tests with full covariance
  
  R = sqrt(2) * [1 1;
                 -1 1];
  d = pml.distributions.gsm(2, 4);
  d.mean = [0; 1];
  d.precision = inv(R * diag([0.5, 1]) * R');
  d.scales = logspace(0.1, 2, 4);
  
  
  test('d.ndims')
  test('d.mean')
  
  test('d.eval([0:0.1:0.9; 0.9:-0.1:0])')
  test('d.sample(10)')  
  pause

  test_probs(d, [0:0.1:0.9; 0.9:-0.1:0]);
  pause

  %% 2D tests + separate precision
  P = d.precision;
  d.precision = {0.5 * P, 0.75 * P, 1 * P, 1.25 * P};
  test_probs(d, [0:0.1:0.9; 0.9:-0.1:0]);
  pause


end

