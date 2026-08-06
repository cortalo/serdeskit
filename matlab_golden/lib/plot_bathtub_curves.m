function plot_bathtub_curves(hax, max_signal, sci_pdf, cci_pdf, isi_and_xtalk_pdf, noise_pdf,jitt_pdf, combined_interference_and_noise_pdf, bin_size)
cursors = d_cpdf(bin_size,max_signal*[-1 1], [1 1]/2);
signal_and_isi_pdf = conv_fct(cursors, sci_pdf);
signal_and_xtalk_pdf = conv_fct(cursors, cci_pdf);
signal_and_channel_noise_pdf = conv_fct(cursors, isi_and_xtalk_pdf);
signal_and_system_noise_pdf = conv_fct(cursors, noise_pdf);
signal_and_system_jitt_pdf = conv_fct(cursors, jitt_pdf);
signal_and_total_noise_pdf = conv_fct(cursors, combined_interference_and_noise_pdf);
%% Added by Bill Kirkland, June 14, 2017
cursors_l = cursors; cursors_l.y(cursors_l.x>0) = 0;
cursors_r = cursors; cursors_r.y(cursors_r.x<0) = 0;
signal_and_total_noise_pdf_l = conv_fct(cursors_l, combined_interference_and_noise_pdf);
signal_and_total_noise_pdf_r = conv_fct(cursors_r, combined_interference_and_noise_pdf);

semilogy(signal_and_isi_pdf.x, abs(cumsum(signal_and_isi_pdf.y)-0.5) ,'r','Disp','ISI', 'parent', hax)
hold on
semilogy(signal_and_xtalk_pdf.x, abs(cumsum(signal_and_xtalk_pdf.y)-0.5) ,'b','Disp','Xtalk', 'parent', hax)
semilogy(signal_and_channel_noise_pdf.x, abs(cumsum(signal_and_channel_noise_pdf.y)-0.5) ,'c','Disp','ISI+Xtalk', 'parent', hax)
semilogy(signal_and_system_noise_pdf.x, abs(cumsum(signal_and_system_noise_pdf.y)-0.5) ,'m','Disp','Jitter, SNR_TX,RL_M, eta_0 noise', 'parent', hax)
semilogy(signal_and_system_jitt_pdf.x, abs(cumsum(signal_and_system_jitt_pdf.y)-0.5) ,'g','Disp','Jitter noise', 'parent', hax)

%% Added by Bill Kirkland, June 14, 2017
% modification allows bathtub curves to cross over and hence one can
% directly read the noise component.
%semilogy(signal_and_total_noise_pdf.x, abs(cumsum(signal_and_total_noise_pdf.y)-0.5) ,'k','Disp','total noise PDF', 'parent', hax)
vbt_l = abs(0.5-cumsum(signal_and_total_noise_pdf_l.y));
vbt_r = fliplr(0.5-(cumsum(fliplr(signal_and_total_noise_pdf_r.y))));
semilogy(signal_and_total_noise_pdf_l.x, vbt_l ,'k','Disp','total noise PDF left', 'parent', hax)
semilogy(signal_and_total_noise_pdf_r.x, vbt_r ,'k','Disp','total noise PDF right', 'parent', hax)

hc=semilogy(max_signal*[-1 -1 1 1], [0.5 1e-20 1e-20 0.5], '--ok');
set(get(get(hc,'Annotation'),'LegendInformation'), 'IconDisplayStyle','off');

ylabel(hax, 'Probability')
xlabel(hax, 'volts')
legend(hax, 'show')
