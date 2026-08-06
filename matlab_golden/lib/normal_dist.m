function pdf = normal_dist(sigma,nsigma,binsize)
pdf.BinSize=binsize;
pdf.Min=-round(nsigma*sigma/binsize);
pdf.x=(pdf.Min:-pdf.Min)*binsize;
pdf.y=exp(-pdf.x.^2/(2*sigma^2+eps));
pdf.y=pdf.y/sum(pdf.y);
