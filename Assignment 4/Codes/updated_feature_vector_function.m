function Energy=updated_feature_vector_function(L,E,S,W,R,extended_boundary_image,N)
e1=average_energy(L'*L,extended_boundary_image,N);

o1=average_energy(L'*E,extended_boundary_image,N);
o2=average_energy(L'*S,extended_boundary_image,N);
o3=average_energy(L'*W,extended_boundary_image,N);
o4=average_energy(L'*R,extended_boundary_image,N);
o5=average_energy(E'*L,extended_boundary_image,N);

e4=average_energy(E'*E,extended_boundary_image,N);

o6=average_energy(E'*S,extended_boundary_image,N);
o7=average_energy(E'*W,extended_boundary_image,N);
o8=average_energy(E'*R,extended_boundary_image,N);
o9=average_energy(S'*L,extended_boundary_image,N);
o10=average_energy(S'*E,extended_boundary_image,N);

e7=average_energy(S'*S,extended_boundary_image,N);

o11=average_energy(S'*W,extended_boundary_image,N);
o12=average_energy(S'*R,extended_boundary_image,N);
o13=average_energy(W'*L,extended_boundary_image,N);
o14=average_energy(W'*E,extended_boundary_image,N);
o15=average_energy(W'*S,extended_boundary_image,N);

e10=average_energy(W'*W,extended_boundary_image,N);

o16=average_energy(W'*R,extended_boundary_image,N);
o17=average_energy(R'*L,extended_boundary_image,N);
o18=average_energy(R'*E,extended_boundary_image,N);
o19=average_energy(R'*S,extended_boundary_image,N);
o20=average_energy(R'*W,extended_boundary_image,N);

e13=average_energy(R'*R,extended_boundary_image,N);

e2 = ((o1+o5)/2);
e3 = ((o6+o10)/2);
e5 = ((o2+o9)/2);
e6 = ((o7+o14)/2);
e8 = ((o3+o13)/2);
e9 = ((o8+o18)/2);
e11 = ((o4+o17)/2);
e12 = ((o11+o15)/2);
e14 = ((o16+o20)/2);
e15 = ((o12+o19)/2);

Energy_vector=[e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14 e15];
Energy=Energy_vector;
end