lams = tb.PoissonProcess(linspace(2.7-0.3*4,2.7+0.3*4,101));
sb = tb.suite_normal(lams,2.7,0.3);
sc = tb.suite_normal(lams,2.7,0.3);
tb.update(sb,[0,2,8,4]);
tb.update(sc,[1,3,1,0]);

bprobs = tb.Suite(0:10,x->tb.margin(tb.poissonPMF,x,sb.hypos))
cprobs = tb.Suite(0:10,x->tb.margin(tb.poissonPMF,x,sc.hypos))

dprobs = bprobs - cprobs
tb.makeCDF(dprobs)
tb.CDF(dprobs,-1)
tb.CDF(dprobs,0) - tb.CDF(dprobs,-1)
1 - tb.CDF(dprobs,0)


using(KernelDensity);

trange = [10:10:1200];
samplesize = 220;

gaptimes_orig = tb.suite_weibull(tb.WaitTime(trange),3,480);
obtimes_orig = tb.suite_observerbias(gaptimes_orig);

sample_z = tb.Suite{tb.WaitTime}(tb.sample(gaptimes_orig,samplesize));
sample_z_biased = tb.suite_observerbias(sample_z);
sample_tmp = tb.sample(sample_z_biased,samplesize);
append!(sample_tmp,[1800,2400,3000]);
k_zb = kde(sample_tmp);
zb_ext = tb.Suite{tb.WaitTime}(trange,h->pdf(k_zb,h));
gaptimes = tb.suite_unbias(zb_ext);


sample_tmp = tb.sample(obtimes_orig,samplesize);
append!(sample_tmp,[1800,2400,3000]);
k_zb = kde(sample_tmp);
obtimes_extended = tb.Suite{tb.WaitTime}(trange,h->pdf(k_zb,h));


obtimes = tb.suite_observerbias(gaptimes);


elapsedtimes = tb.suite_waittimes(obtimes);
elprior = deepcopy(elapsedtimes);
tb.update(elapsedtimes,(15,2.0/60.0));
waittimes = obtimes - elapsedtimes;
tb.remove_negatives(waittimes);

tb.plot_cdf(gaptimes,"k")
tb.plot_cdf(obtimes,"b")
tb.plot_cdf(elprior,"c")
tb.plot_cdf(elapsedtimes,"g")
tb.plot_cdf(waittimes,"r")

#---------#----#
#---> prior distrib of time since gap start weighted by gap prob and gap length
#-----> posterior based on waiting passengers
#      <--# remaining time = gap prob * gap length - time since gap start * gap length

#el_unbiased = tb.suite_waittimes(gaptimes);
#el_un_prior = deepcopy(el_unbiased);
#tb.update(el_unbiased,(15,2.0/60.0));
#wait_unbiased = gaptimes - el_unbiased;
#tb.remove_negatives(wait_unbiased);
#wait2 = tb.suite_observerbias(wait_unbiased);
#tb.plot_cdf(wait2,"k")
