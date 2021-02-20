function test_suite = test_density
  initTestSuite;
end

function [f, g] = log_plus_log_grad_mu(x, d, sizex)
  x = reshape(x, sizex);
  f = d.log(x);
  g = d.log_grad_mu(x);
end

function test_all
  R = sqrt(2) * [1 1;
                 -1 1];
  d = pml.distributions.gsm(2, 4);
  d.mean = [0; 1];
  d.precision = inv(R * diag([0.5, 1]) * R');
  d.scales = logspace(0.1, 2, 4);
  d.mean = 0;
  d.precision = 100;
  d
  x = d.sample(1);
  
  % d.mu, d.precision
  % d.scales, d.weights
  % scatter(x(1,:), x(2,:))
  % [Ak A Bk B Ck C Dk D] = d.ho_derivatives(x)
  
  % x(2) = 0;
  grad_err = pml.numerical.checkgrad(@log_plus_log_grad_mu, x(:), 1.e-6, d, size(x))
end