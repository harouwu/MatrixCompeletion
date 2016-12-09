%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [loss] = computeLoss(R, X, W, H, Y, lambda, Omega)
    
    P = X * W * H * Y';
    R(Omega) = R(Omega) - P(Omega);
    loss = norm(R,'fro')^2 + lambda * norm(W,'fro')^2 + lambda * norm(H, 'fro')^2;

end