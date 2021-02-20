function test_suite = test_density
  initTestSuite;
end

function [f, g] = f_plus_grad(x, d, sizex)
  x = reshape(x, sizex);
  [Ak A Bk B Ck C Dk D] = d.ho_derivatives(x);%, pause
  f = A; g = B;
  % f = B; g = C;
  % f = C; g = D;
  
  i = 1;
  % f = B; g = zeros(sizex); f = f(i); g(i) = C(i);
  % f = C; g = zeros(sizex); f = f(i); g(i) = D(i);
  % f = C; g = D; f = f(1);
  
  
  % f = f(1);
  % g = g(1,:)';
  
  % f = sum(sum(f));
  % g = sum(g,1)';
  
  % f, g, pause
  
end

function test_all
  R = sqrt(2) * [1 1;
                 -1 1];
  d = pml.distributions.gsm(2, 4);
  d.mean = [0; 1];
  d.precision = inv(R * diag([0.5, 1]) * R');
  d.scales = logspace(0.1, 2, 4);
  x = d.sample(1);
  
  % d.mu, d.precision
  % d.scales, d.weights
  % scatter(x(1,:), x(2,:))
  % [Ak A Bk B Ck C Dk D] = d.ho_derivatives(x)
  
  d = pml.distributions.gsm(1, 5);
  d.precision = 1./100;
  d.scales = exp(-4:2:4);
  x = d.sample(1);
  
  % x(2) = 0;
  grad_err = pml.numerical.checkgrad(@f_plus_grad, x(:), 1.e-6, d, size(x))
end