#!/opt/anaconda/bin/python -u
import sys, os, sklearn, copy, shutil
import numpy as np
import pandas as pd
from pyfitit import *


cache = utils.Cache(folder=f'cache', debug=False)


def getNewTPRTemp():
    new_data = pd.read_csv(f'V_TPR_all.txt', sep='\t')
    temp =[float(t) for t in new_data.columns[1:-1]]
    return np.array(temp)


def loadData2(crParams, vParams, mergePreedgeByEfermi, toDel, shift):
    if isinstance(toDel, str): assert toDel in ['all V', 'all Cr']
    sample_cr = readSample('sample Cr')
    sample_van = readSample('sample V')
    if not crParams['reg']: sample_cr.params['formal charge'] = np.round(sample_cr.params['formal charge'])
    if not vParams['reg']: sample_van.params['formal charge'] = np.round(sample_van.params['formal charge'])
    for spType in ['xanes', 'xanesWoPre-edge']:
        sample_van.shiftEnergy(shift, spType=spType, inplace=True)
        sample_van.changeEnergy(sample_cr.getEnergy(spType), spType=spType, inplace=True)
    pec_cr = np.median(sample_cr.params['pe center'])
    pec_van = np.median(sample_van.params['pe center'])
    shift_pe = shift if mergePreedgeByEfermi else pec_cr - pec_van
    sample_van.shiftEnergy(shift_pe, spType='pre-edge', inplace=True)
    sample_van.changeEnergy(sample_cr.getEnergy('pre-edge'), spType='pre-edge', inplace=True)

    sample_van.changeEnergy(sample_cr.getEnergy('exafs'), spType='exafs', inplace=True)
    if 'formal CN' in sample_cr.labelMaps: sample_cr.decode(label='formal CN')
    if 'formal charge' in sample_van.labelMaps: sample_van.labelMaps['formal charge'][6] = 5
    sample = sample_cr.copy()

    def delParams(s:ML.Sample):
        for pn in s.paramNames:
            if pn not in ['formal charge', 'formal CN', 'Short name']:
                s.delParam(pn)

    delParams(sample)
    delParams(sample_van)
    if toDel == 'all Cr': sample = sample_van
    elif toDel != 'all V':
        sample.unionWith(sample_van, inplace=True)
        if toDel is not None: sample.delRowByName(toDel, inplace=True)
    if toDel in ['all Cr', 'all V']:
        for name in toDelAnyway:
            if name in sample.params[sample.nameColumn]:
                sample.delRowByName(name, inplace=True)
    sample.addParam(paramName='isCr', paramData=np.isin(sample.params['Short name'], sample_cr.params['Short name']).astype(float))
    sample1 = sample.copy()
    for i in range(len(getNewTPRTemp())): sample1.delRowByName(f'TPR_{i}')
    rename(sample)
    sample.delParam('isCr')
    addDescrFunc(sample1, usePrebuiltData=False)
    sample = addDescrFunc(sample, usePrebuiltData=True)
    return sample


def rename(unionSample:ML.Sample):
    cr_data = pd.read_excel(f'Spectral Library.xlsx', sheet_name='Cr', skiprows=1, index_col='Short name')
    v_data = pd.read_excel(f'Spectral Library.xlsx', sheet_name='V', skiprows=1, index_col='Short name')
    for i in range(len(unionSample)):
        old_name = unionSample.getName(i)
        if old_name.startswith('TPR_'):
            new_name = old_name
        else:
            s = cr_data if unionSample.params.loc[i,'isCr'] else v_data
            new_name = s.loc[old_name,'Short name 09.05.2024']
        unionSample.params.loc[i,unionSample.nameColumn] = new_name


def loadData(crParams, vParams, mergePreedgeByEfermi, toDel, shift):
    return cache.getFromCacheOrEval('united_sample', evalFunc=lambda: loadData2(crParams, vParams, mergePreedgeByEfermi, toDel, shift), dependData=(crParams, vParams, mergePreedgeByEfermi, toDel, toDelAnyway,shift))


def addDescrFunc(sample, usePrebuiltData=True):
    fast = True
    if usePrebuiltData or fast:
        debugEfermi, debug1st_peak = {}, {}
    else:
        debugEfermi = {'plotFolder':f'results/efermi', 'maxNumToPlot':1000}
        debug1st_peak = {'plotFolder':f'results/1st_peak', 'maxNumToPlot':1000}
    sample = descriptor.addDescriptors(sample, [
        dict(type='efermi', spType='xanesWoPre-edge', **debugEfermi),
        dict(type='area', columnName='pe area', spType='pre-edge'),
        dict(type='center', columnName='pe center', spType='pre-edge'),
        dict(type='1st_peak', spType='xanesWoPre-edge', **debug1st_peak),
        dict(type='pca', energyInterval=[5980,6120], spType='xanes', usePrebuiltData=usePrebuiltData, count=3, fileName=f'cache/prebuilt_pca.pkl'),
        dict(type='scaled_pca', energyInterval=[5980,6120], spType='xanes', usePrebuiltData=usePrebuiltData, count=3, fileName=f'cache/prebuilt_scaled_pca.pkl'),
        dict(type='value', spType='xanes', energies=[5995, 6010, 6042]),
        dict(type='separable', labels=['formal charge', 'formal CN'], normalize=True, pairwiseTransformType='binary', spType='xanes', features='xanes spectra', energyInterval=[5980,6100], debugFolder='results/separable_debug'),
    ], inplace=False)[0]

    classical_features = sorted(list(set(sample.paramNames) - {'Short name','isCr'} - set(sample.labels)))

    # baggingParams=dict(max_samples=0.1, max_features=1.0, n_estimators=10, random_state=0),
    sample = descriptor.addDescriptors(sample, [
        dict(type='best_linear', label='formal charge', features=['1st_peak_e', '1st_peak_i'], columnName='BL1stCharge', cv_parts=20,  usePrebuiltData=usePrebuiltData, fileName=f'cache/?'),
        dict(type='best_linear', label='formal charge', features=["value_5995", "value_6010", "value_6042"], columnName='BLvaluesCharge', cv_parts=20, usePrebuiltData=usePrebuiltData, fileName=f'cache/?'),
        dict(type='best_linear', label='formal CN', features=["value_5995", "value_6010", "value_6042"], columnName='BLvaluesCN', cv_parts=20, usePrebuiltData=usePrebuiltData, fileName=f'cache/?'),
    ], inplace=False)[0]

    sample = descriptor.addDescriptors(sample, [
        dict(type='tsne', columnName='tsne_pca', energyInterval=[5980, 6080], spType='xanes', preprocess='pca 5', usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_tsne.pkl'),
        dict(type='tsne', columnName='tsne_pca_scale', energyInterval=[5980, 6080], spType='xanes', preprocess='scaler and pca 5', usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_tsne_scale.pkl'),
        dict(type='tsne', columnName='tsne_values', features=["value_5995", "value_6010", "value_6042"], usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_tsne_values.pkl'),
    ], inplace=False)[0]
    sample = descriptor.addDescriptors(sample, [
        dict(type='pls', label='formal CN', spType='xanes', energyInterval=[5980,6080], usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_pls_xanes_CN.pkl'),
        dict(type='pls', label='formal charge', features=classical_features, usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_pls_features_charge.pkl'),
        dict(type='pls', label='formal CN', features=classical_features, usePrebuiltData=usePrebuiltData, count=2, fileName=f'cache/prebuilt_pls_features_CN.pkl'),
    ], inplace=False)[0]
    sample.features = sorted(list(set(sample.paramNames) - {'Short name'} - set(sample.labels)))
    return sample


def plotScatter(method, reg, paper_pics, mergePreedgeByEfermi, toDel, shift):
    assert method in ['Extra Trees', 'RidgeCV']
    import itertools
    folder = f'results/united scatter {method}'
    folder += '_efermiMerge' if mergePreedgeByEfermi else '_peCenterMerge'
    singleSample = loadData(crParams=dict(reg=reg, delCr=False), vParams=dict(references='noCl', unknown='all', reg=reg), mergePreedgeByEfermi=mergePreedgeByEfermi, toDel=toDel, shift=shift)
    known, unknown = singleSample.splitUnknown('formal CN')
    unknown = unknown.takeRowsByName(names=['VTi0', 'VTi1', 'VTi2'])
    if method == 'Extra Trees':
        model_class, model_regr = sklearn.ensemble.ExtraTreesClassifier(class_weight='balanced'), None
    elif method == 'RidgeCV':
        model_class = ML.Normalize(sklearn.linear_model.LogisticRegression(), xOnly=True)
        model_regr = ML.Normalize(sklearn.linear_model.RidgeCV(), xOnly=True)
    if paper_pics:
        folder = f'results/united scatter for paper' if method=='Extra Trees' else f'results/united scatter for paper linear'
        # lab_desc = ['formal charge', 'pls1_features_formal CN', 'pls2_features_formal CN'], ['formal CN', 'pls2_features_formal CN', 'pls2_features_formal charge'], ['formal charge', 'pe center', 'pe area'], ['formal CN', 'pe center', 'pe area']
        lab_desc = ['formal charge', 'pe center', 'pe area'], ['formal CN', 'pls1_features_formal CN', 'pls2_features_formal CN'], ['formal charge', 'pls1_features_formal charge', 'value_5995'], ['formal CN', 'efermi_slope', 'value_6042'],
        # lab_desc = ['formal charge', 'pe center', 'pe area'], ['formal CN', 'pe center', 'pe area']
        if not reg: known.params['formal CN'] = np.round(known.params['formal CN'])
        for label, desc1, desc2 in lab_desc:
            descriptor.plotDescriptors2d(known.params, descriptorNames=[desc1, desc2], labelNames=[label], labelMaps=singleSample.labelMaps, folder_prefix=folder, unknown=unknown.params, textColumn=singleSample.nameColumn, cv_count=50, additionalMapPlotFunc=None, cmap='viridis', model_class=model_class, model_regr=model_regr, textsize=7, linewidth=0.5, markersize=15, fileExtension='.svg')
    else:
        labels=['formal charge', 'formal CN']
        for fs in itertools.combinations(singleSample.features, 2):
            additionalMapPlotFunc = None
            if set(fs) == {'pe area', 'pe center'}:
                fs = ['pe center', 'pe area']
            descriptor.plotDescriptors2d(known.params, descriptorNames=fs, labelNames=labels, labelMaps=singleSample.labelMaps, folder_prefix=folder, unknown=unknown.params, textColumn=singleSample.nameColumn, cv_count=50, additionalMapPlotFunc=additionalMapPlotFunc, cmap='viridis', model_class=model_class, model_regr=model_regr)


def directPrediction(interval, reg, mergePreedgeByEfermi, toDel, folder=None, byMixture=False, shift=None):
    predictByDescr = interval is None
    if predictByDescr: interval = [5980,6120]
    singleSample:Sample = loadData(crParams=dict(reg=reg, delCr=False), vParams=dict(references='noCl', unknown='all', reg=reg), mergePreedgeByEfermi=mergePreedgeByEfermi, toDel=toDel, shift=shift)
    singleSample.limit(interval, spType='xanes', inplace=True)
    # s0 = singleSample.getSpectrum(name='TPR0',spType='xanes')
    # s1 = singleSample.getSpectrum(name='TPR_0',spType='xanes')
    # plotting.plotToFile(s0.x,s0.y,'TPR0', s1.x,s1.y,'TPR_0', fileName='results/debug.png')
    # по качеству для formal charge full всегда лучше
    # singleSample.changeEnergy(np.linspace(*interval, 100), spType='xanes', inplace=True)
    folder0 = folder
    if folder is None:
        folder = f'results/direct_pred_{interval[1]}'
        if reg: folder += '_reg'
        folder += '_efermiMerge' if mergePreedgeByEfermi else '_peCenterMerge'
        random_state = 0
    else:
        random_state = None
    known, unknown = singleSample.splitUnknown(columnNames='formal charge')
    labels = ['formal charge', 'formal CN']  # singleSample.labels
    all_features = singleSample.features
    for n in singleSample.paramNames:
        if n not in singleSample.features and n != singleSample.nameColumn and n not in singleSample.labels:
            singleSample.delParam(n)
    for label in labels:
        if predictByDescr:
            predictBy = {
                 # 'human': ['pe area', 'pe center', "efermi_e", "efermi_slope", "1st_peak_e"],
                 'pca': ["pca1", "pca2", "pca3"],
                 'scaledPca': ["scaled_pca1", "scaled_pca2", "scaled_pca3"],
                 'pe': ['pe area', 'pe center'],
                 'separable': ['separable_formal charge', 'separable_formal CN'],
                 'indepBogdan': ['pls1_features_formal charge', 'tsne_pca2_xanes', 'value_6042', 'pe area'],
                 }
            predictBy['tsne'] = [f for f in all_features if f.startswith('tsne_')]
            predictBy['pls'] = [f for f in all_features if f.startswith('pls')]
            predictBy['values'] = [f for f in all_features if f.startswith('value_')]
            predictBy['BL'] = [f for f in all_features if f.startswith('BL')]
            if label == 'formal CN':
                predictBy['indepCr'] = ['efermi_slope', 'pls1_features_formal CN', 'tsne_values1_features', 'pls2_features_formal charge']
                predictBy['indepV'] = ['BLvaluesCN', 'pls2_features_formal charge', 'value_5995']
            elif label == 'formal charge':
                predictBy['indepCr'] = ['BLvaluesCharge', 'pls1_features_formal charge', 'value_6010']
                predictBy['indepV'] = ['BL1stCharge', 'pls1_features_formal charge', 'pe center', 'efermi_e', 'value_6010']
            else: assert False
        else: predictBy = {'xanes': ['xanes spectra_d_0,1']}
        for name,fs in predictBy.items():
            explanationParams = None
            # if fs[0] == 'xanes spectra' and folder0 is None:
            #     explanationParams = {'explainFor':['V.5y.1']}
            if byMixture:
                folder1 = f'{folder}/mix by {name}' if folder0 is None else folder
                makeMixtureParams = {'componentCount':2, 'addDescrFunc':addDescrFunc, 'componentNameColumn':singleSample.nameColumn}
                # мы может построить дескрипторы только при наличии всех labels
                descriptor.directPrediction(known, fs, labels, labelMaps=None, folder=folder1, cv_count=4, repForStdCalc=3, textColumn=singleSample.nameColumn, unknown_sample=unknown, plot_diff=True, explanationParams=explanationParams, random_state=random_state, makeMixtureParams=makeMixtureParams)
            else:
                folder1 = f'{folder}/single by {name}' if folder0 is None else folder
                descriptor.directPrediction(known, fs, [label], labelMaps=singleSample.labelMaps, folder=folder1, cv_count=50, repForStdCalc=1, textColumn=singleSample.nameColumn, unknown_sample=unknown, plot_diff=True, explanationParams=explanationParams, random_state=random_state)


def tryAllMix(interval, runCV, spType, mergePreedgeByEfermi, toDel, folder=None, run1comp=True, shift=None):
    labels = ['formal charge', 'formal CN']
    if spType == 'default': spType = 'xanes'
    folder0 = folder
    if folder is None:
        folder = f'results/LCF {spType} {interval[1]}'
        folder += '_efermiMerge' if mergePreedgeByEfermi else '_peCenterMerge'
    interval0 = copy.deepcopy(interval)
    singleSample:Sample = loadData(crParams=dict(reg=False, delCr=False), vParams=dict(references='noCl', unknown='all', reg=False), mergePreedgeByEfermi=mergePreedgeByEfermi, toDel=toDel, shift=shift)
    singleSample.limit(interval0, inplace=True, spType=spType)
    known, unknown = singleSample.splitUnknown(columnNames='formal charge')
    commonParams = dict(mixtureTrysCount='all combinations of singles', optimizeConcentrationsTrysCount=10, labelNames=labels,  fileNamePostfix='', spectraFolderPostfix='', labelMapsFolderPostfix='', labelBounds=None, labelMaps=singleSample.labelMaps, componentNameColumn=singleSample.nameColumn, calcDescriptorsForSingleSpectrumFunc=lambda s: None, randomSeed=0, makeMixtureOfSpectra=None, plotSpectrumType='all', maxSpectraCountToPlot=10, customPlotter=None)
    if runCV:
        unknownCharacterization = {'type': 'spectrum', 'spType': spType}
        singleSample1 = singleSample.limit(interval0, inplace=False, spType=spType)
        if run1comp:
            mixture.tryAllMixturesCV(componentCount=1, singleComponentData=singleSample1, optimizeConcentrations=False, folder=f'{folder} 1/CV', unknownCharacterization=unknownCharacterization, **commonParams)
        folder1 = folder if folder0 is not None and not run1comp else f'{folder} 2/CV'
        mixture.tryAllMixturesCV(componentCount=2, singleComponentData=singleSample1, optimizeConcentrations=True, folder=folder1, unknownCharacterization=unknownCharacterization, **commonParams)
        return
    for ii in range(len(unknown)):
        name = unknown.params.loc[ii,unknown.nameColumn]
        # выборочные
        # if name not in ['VTi0', 'VTi1', 'VTi2']: continue

        # все кроме TPR
        #if name.startswith('TPR_'): continue

        # выборочные TPR
        #if not name.startswith('TPR_'): continue
        # от 0 до 122
        #if not ( 80<=int(name[4:])<=122 ): continue

        exp = unknown.getSpectrum(ii, spType='all types')
        
        # name = 'V.III.6o.1b'
        # exp = singleSample.getSpectrum(name='V.III.6o.1b', spType='all types')
        # known.delRowByName('V.III.6o.1b', inplace=True)

        i = len(exp[spType].y)-1
        while exp[spType].y[i] == exp[spType].y[-1]: i = i-1
        i = i+1
        if i != len(exp[spType].y)-1:
            interval = [interval[0], exp[spType].x[i]]
            exp[spType] = exp[spType].limit(interval)
            print(f'Spectrum {name} has shorter interval: {interval}')
            known1 = known.limit(interval, inplace=False, spType=spType)
        else: known1 = known
        unknownCharacterization = {'type':'spectrum', 'spectrum':exp[spType], 'spType':spType}
        commonParams['singleComponentData'] = known1
        commonParams['unknownCharacterization'] = unknownCharacterization
        commonParams['unknownSpectrumToPlot'] = exp
        if run1comp:
            mixture.tryAllMixtures(componentCount=1, optimizeConcentrations=False, folder=f'{folder} 1/{name}', **commonParams)
        folder1 = f'{folder}/{name}' if folder0 is not None and not run1comp else f'{folder} 2/{name}'
        mixture.tryAllMixtures(componentCount=2, optimizeConcentrations=True, folder=folder1, **commonParams)
        # exit(0)


def resultSummaryDescr(mergePreedgeByEfermi, toDel, shift):
    postfix = '_reg'
    postfix2 = '_efermiMerge' if mergePreedgeByEfermi else '_peCenterMerge'
    postfix += postfix2
    crParams = dict(reg=True, delCr=False)
    vParams = dict(references='noCl', unknown='all', reg=True)
    singleSample:Sample = loadData(crParams=crParams, vParams=vParams, mergePreedgeByEfermi=mergePreedgeByEfermi, toDel=toDel, shift=shift)
    for i in range(len(getNewTPRTemp())): singleSample.delRowByName(f'TPR_{i}')
    _, unknown = singleSample.splitUnknown(columnNames=['formal charge'])
    settings = [dict(type='LCF', folder=f'LCF xanes 6120{postfix2} 2', prefix='long '),
                dict(type='direct', folder=f'direct_pred_6120{postfix}/single by xanes', prefix=f'MLs long '),
                # dict(type='direct', folder=f'direct_pred_6050{postfix}/single by xanes', prefix=f'MLs short '),
    ]
    # names = ['BL', 'indepCr', 'indepV', 'indepBogdan', 'pca', 'scaledPca', 'pls', 'tsne', 'values', 'pe']
    names = ['pca', 'pls', 'tsne', 'pe']
    for fs in names:  # names:
        settings += [dict(type='direct', folder=f'direct_pred_6120{postfix}/single by {fs}', prefix=f'MLs {fs} ')]
    # for fs in names:
    #     settings += [dict(type='direct', folder=f'direct_pred_6120{mixpostfix}/mix by {fs}', prefix=f'MLm {fs} ')]
    # if toDel == 'all Cr': bf = 'V only'
    # elif toDel == 'all V': bf = 'Cr only'
    # else: bf = 'Cr+V'
    descriptor.resultSummary(sample=singleSample, labels=['formal CN', 'formal charge'], unknownNames=unknown.params[unknown.nameColumn], baseFolder=f'results', wrapXTickLabelLength=100, figsize=(13,13), settings=settings, postfix=postfix, fileExtension='.svg', criticalQuantileIndivPlot=1)


def combineAllTPR():
    parentFolder = 'results'
    labels = ['formal charge', 'formal CN']
    # считаем доверительный интервал
    for label in labels:
        print(f'Label {label}')
        d = pd.read_excel(f'{parentFolder}/summary_reg_efermiMerge/known {label} pred-true.xlsx', index_col='Unnamed: 0')
        for method in d.columns:
            std = np.std(d[method])
            print(f'STD for {method} = {std}')
            if label == 'formal CN' and method == 'LCF2 long ':
                lcf_std = std
    xi1 = np.random.normal(loc=0, scale=lcf_std, size=10000)
    xi2 = np.random.normal(loc=0, scale=lcf_std, size=10000)
    p_value = np.sum(xi1+1<xi2)/len(xi1)
    print(f'p-value for LCF2 increase by 1 = {p_value}')

    folder = f'{parentFolder}'
    T = getNewTPRTemp()
    unk_names = [f'TPR_{i}' for i in range(len(T))]
    d = {l:pd.DataFrame(index=unk_names) for l in labels}
    for parentFolder in ['direct_pred_6100_reg_efermiMerge', 'direct_pred_6120_reg_efermiMerge']:
        for subFolder in os.listdir(f'{folder}/{parentFolder}'):
            if subFolder.startswith('mix '): continue
            _, unk = descriptor.directPredictionParser(f'{folder}/{parentFolder}/{subFolder}')
            method = 'ML '+subFolder[subFolder.find(' by ')+4:]
            for l in labels:
                for uname in unk_names: d[l].loc[uname,method] = unk.loc[uname,l]
    lcf = mixture.tryAllMixturesParser(f'{folder}/LCF xanes 6120_efermiMerge 2')
    for l in labels:
        for uname in unk_names: d[l].loc[uname,'LCF2'] = lcf.loc[uname,f'{l} prediction']
    for label in labels: 
        os.makedirs(f'{folder}/TPR summary', exist_ok=True)
        d[label].to_excel(f'{folder}/TPR summary/{label}.xlsx')
        toPlot = tuple()
        for im, method in enumerate(d[label].columns):
            pred = [d[label].loc[f'TPR_{i}', method] for i in range(len(T))]
            fmt = '-' if im<10 else ':'
            toPlot += (T,pred,{'label':method, 'fmt':fmt})
        plotToFile(*toPlot, fileName=f'{folder}/TPR summary/{label}.svg')


if __name__ == "__main__":
    # toDelAnyway = []
    toDelAnyway = ['Cr', '0.6o.1', 'V2O4', 'NaV2O5', 'VNOEt3']
    shift = 522.3
    toDel = toDelAnyway  # 'all V'  'all Cr' or toDelAnyway or list of names
    plotScatter(method='Extra Trees', reg=False, paper_pics=True, mergePreedgeByEfermi=True, toDel=toDel, shift=shift)

    tryAllMix(interval=[5970, 6120], runCV=True, spType='xanes', mergePreedgeByEfermi=True, toDel=toDel, shift=shift)
    tryAllMix(interval=[5970, 6120], runCV=False, spType='xanes', mergePreedgeByEfermi=True, toDel=toDel, shift=shift)

    # prediction by xanes long
    directPrediction(interval=[5980, 6120], reg=True, mergePreedgeByEfermi=True, toDel=toDel, byMixture=False, shift=shift)
    # prediction by descriptors
    directPrediction(interval=None, reg=True, mergePreedgeByEfermi=True, toDel=toDel, byMixture=False, shift=shift)
    # prediction by xanes for TPR serie
    directPrediction(interval=[5980, 6100], reg=True, mergePreedgeByEfermi=True, toDel=toDel, byMixture=False, shift=shift)

    resultSummaryDescr(mergePreedgeByEfermi=True, toDel=toDel, shift=shift)
    combineAllTPR()


