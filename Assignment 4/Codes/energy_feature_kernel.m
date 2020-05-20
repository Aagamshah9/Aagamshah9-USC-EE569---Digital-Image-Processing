function E=energy_feature_kernel(r_map,row,col,n)
K=extended_boundary_r_map(r_map,row,col,n);
for i = ((n-1)*0.5)+1:row+((n-1)*0.5)
    for j = ((n-1)*0.5)+1:col+((n-1)*0.5)
        energy_pixel = 0;
        for x = -((n-1)*0.5):((n-1)*0.5)
            for y = -((n-1)*0.5):((n-1)*0.5)
                energy_pixel = energy_pixel + (abs(K(i+x,j+y)));
            end
        end
        E(i-((n-1)*0.5),j-((n-1)*0.5)) = energy_pixel;
    end
end
end