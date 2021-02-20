function img_sz = get_img_sz(img, filters)
% Haichao Zhang
% 2011-8-5 15:26:57

for i = 1:numel(filters)
%     img_sz{i} = size(conv2(img, filters{i}, 'valid'));
    img_sz{i} = size(my_conv2(img, filters{i}));
%    img_sz{i} = size(obs_for(img, filters{i}, 1));
end