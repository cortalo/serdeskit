function [ pdf ] = get_pdf_from_sampled_signal( input_vector, L, BinSize ,FAST_NOISE_CONV)
% Create PDF from interference vector using successive delta-set convolutions.
%   input_vector = list of values of samples
%   return
%   pdf.x
%   pdf.y
%   pdf.vec
%   pdf.bin
if ~exist('FAST_NOISE_CONV','var')
    FAST_NOISE_CONV=0;
end
if max(input_vector) > BinSize
    input_vector=input_vector(abs(input_vector)>BinSize);
end
% for i = 1:length(input_vector)
%    if abs(input_vector(i)) < BinSize , input_vector(i)=0; end
%end

input_vector(abs(input_vector)<BinSize) = 0;
b=sign(input_vector);
[input_vector,index]=sort(abs(input_vector),'descend');
input_vector=input_vector.*b(index);
if FAST_NOISE_CONV
    sig_res=norm(input_vector(find(abs(input_vector)<.001,1)+1:end));
    res_pdf= normal_dist(sig_res,5,BinSize);
    input_vector=input_vector(1:find(abs(input_vector)<.001,1));
end
%% Equation 93A-39 %%
values = 2*(0:L-1)/(L-1)-1;
prob = ones(1,L)/L;

%% Initialize pdf to delta at 0
pdf=d_cpdf(BinSize, 0, 1);
empty_pdf=pdf;
for k = 1:length(input_vector)
    %     pdfn=d_cpdf(BinSize, abs(input_vector(k))*values, prob);
    pdfn=Init_PDF_Fast(empty_pdf, abs(input_vector(k))*values, prob);
    pdf=conv_fct(pdf, pdfn);
end
if FAST_NOISE_CONV
    %     pdf=conv_fct(pdf,res_pdf);
    pdf=conv_fct_TEST(pdf,res_pdf);
end
