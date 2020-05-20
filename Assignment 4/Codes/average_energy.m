function E=average_energy(band_filter,extended_boundary_image,N)
E=0;
for i=3:N+2
    for j=3:N+2
        avg=0;
        for k = -2:2
            for l = -2:2
                avg = avg + (extended_boundary_image(i+k,j+l)*band_filter(k+3,l+3));
            end
        end
        r_map(i-2,j-2)=avg;
        E = E + ((1/(N*N))*((abs(r_map(i-2,j-2)))));
    end
end
end