function test_suite = test_filter
  initTestSuite;
end

function test_show
  
  % load '../../@gsm_pairwise_mrf/test/trainingImages_double.mat'
  % for i = 1:length(X), X{i} = X{i} / 255; end
  
  load 'results/trainingImages_zeromean_unitvariance_double.mat'
  
  npatches = 1; imdims = [30 30];
  % npatches = 2500; imdims = [30 30];
  patches = pml.support.random_patches(X, imdims, npatches);
  
  % load('whitening_original_30x30.mat');
  load('whitening_zeromean_unitvariance_30x30.mat');
  % load('whitening_wf_30x30.mat');
  
  patches_w = W * patches;
  img = pml.support.assemble_images(patches, imdims, ceil(sqrt(40*npatches)));
  img_w = pml.support.assemble_images(patches_w, imdims, ceil(sqrt(40*npatches)));
  
  % mean(img(:,1:10)), var(img(:,1:10))
  % mean(img_w(:,1:10)), var(img_w(:,1:10))
  
  figure(1), clf, colormap(gray(256))
  subplot1(1,2)
  subplot1(1), imagesc(img), axis image off, colorbar
  subplot1(2), imagesc(img_w), axis image off, colorbar
  % return
  
  figure(2), clf, colormap(gray(256))
  % imagesc(W), axis image, colorbar
  imagesc(reshape(W(435,:), [30 30])), axis image, colorbar
  
end

function test_filters

  % loads cell array X of 40 trainings images
  % load '../../@gsm_pairwise_mrf/test/trainingImages_double.mat'
  load 'results/trainingImages_zeromean_unitvariance_double.mat'
  
  % for i = 1:length(X), X{i} = X{i} / 255; end
  
  % #patches per image
  npatches = 100; imdims = [30 30];
  % npatches = 200; imdims = [30 30];
  patches = pml.support.random_patches(X, imdims, npatches);
  % % subtract mean from each image and from each variable
  % patches = bsxfun(@minus, patches, mean(patches));
  % patches = bsxfun(@minus, patches, mean(patches,2));
  size(patches)
  
  % [Kwhiten, A] = whiteningFilter(patches', imdims);
  A = zerophase_whitening(patches);
  figure(1), clf, colormap(gray(256))
  %subplot(2,1,1)
  imagesc(A), axis image, colorbar
  % subplot(2,1,2), imagesc(Kwhiten), axis image, colorbar
  % save(sprintf('whitening_%dx%d.mat', imdims(1), imdims(2)), 'Kwhiten', 'A')
  return
  
  E = cov(patches');
  figure(1), clf, colormap(gray(256))
  subplot(2,1,1), imagesc(E), axis image, colorbar
  
  % E = E + 1.e-8*eye(size(E));
  
  [V,D] = eig(E);
  W = D^(-1/2) * V';
  A = D^( 1/2) * V';
  % save(sprintf('A_%dx%d.mat', imdims(1), imdims(2)), 'A')
  
  patches_whitened = W * patches;
  E_whitened = cov(patches_whitened');
  subplot(2,1,2), imagesc(E_whitened), axis image, colorbar
  
  
  
  J = randn(prod(imdims), 10);
  J = bsxfun(@minus, J, mean(J));
  rank(A), mean(A'*J)
  
  figure(2), clf, colormap(gray(256))
  imagesc(A), axis image, colorbar
  
end

% code by yair weiss
function [Kwhiten,A]=whiteningFilter(xs,siz)
  % calculate the zero phase whitening filter using PCA
  % xs is a npatches*npixels _in_patch matrix where each row is a patch
  % siz is the size of the patch (e.g. [5 5])
  cov=xs'*xs;
  [uu,ss,vv]=svd(cov);
  dd=diag(ss);
  D=diag(sqrt(1./dd));
  A=uu*D*uu';
  Kwhiten=reshape(A(round(prod(siz)/2),:),siz);
  Kwhiten=Kwhiten';
  Kwhiten=Kwhiten-mean(Kwhiten(:));
   Kwhiten=Kwhiten/norm(Kwhiten(:));
end

% code by stefan roth
function [W] = zerophase_whitening(X, option)
  
  if (nargin > 1)
    switch (option)
     case 1
      C = X * X' / (size(X, 2) - 1);
      C = 0.5 * (C + C');
     case 'cov'
      C = X;
     otherwise
      C = cov(X');
    end
  else
    C = cov(X');
  end
  
  % C = X;
  [U, S, V] = svd(C,0);
  [S, I] = sort(diag(S), 1, 'descend');
  S = diag(S);
  U = U(:, I);

  W = U * diag(1 ./ sqrt(diag(S))) * U';
  % DW = U * sqrt(S);
end