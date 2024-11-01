# srwg_risk_toolbox can be found at
# https://github.com/GNS-Science/srwg-risk-toolbox

from srwg_risk_toolbox import *

hcurves_file = 'hcurves_mini.hdf5'

### access the metadata for the hazard curves
with h5py.File(hcurves_file, 'r') as hf:
    hf_vs30_list = list(hf['metadata'].attrs['vs30s'])
    sites = pd.DataFrame(ast.literal_eval(hf['metadata'].attrs['sites']))
    hf_site_list = list(sites.index)
    imtls = ast.literal_eval(hf['metadata'].attrs['acc_imtls'])
    hf_imt_list = list(imtls.keys())
    hf_period_list = [period_from_imt(imt) for imt in hf_imt_list]
    quantiles = hf['metadata'].attrs['quantiles']

    hcurves = hf['hcurves']['hcurves_stats'][:]


### specify the parameters of interest
site_list = ['Auckland','Wellington']
sc_list = ['IV']
imt_list = ['SA(0.5)']
hf_i_metric = 0  # index corresponds to ['mean'] + quantiles

### iterate through parameters
for site in site_list:
    for sc in sc_list:
        vs30 = choose_representative_vs30(sc)

        for imt in imt_list:
            hf_i_site = hf_site_list.index(site)
            hf_i_imt = hf_imt_list.index(imt)
            hf_i_vs30 = hf_vs30_list.index(vs30)

            hcurve = hcurves[hf_i_vs30, hf_i_site, hf_i_imt, :, hf_i_metric]
            imtl = imtls[imt]

            print(f'\n{site}, SC:{sc}, {imt}:')
            print(f'\tAPoEs:\n{hcurve}')
            print(f'\tIMTLs:\n{imtl}')