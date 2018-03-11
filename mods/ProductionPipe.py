import sys
sys.path.append("../")
import mods
from sklearn.pipeline import Pipeline

def CreatePipeline(jobs = -1):
    to_drop = ['Date','Time','Open','High','Low','Close','Volume',"Period_4_Lookforward_p1","Period_4_Lookforward_p2","Period_20_Close_Std"]
    to_bool = []

    layer_1 = [('pass',mods.ColumnDropperTransformer())]
    layer_2 = [('pass',mods.ColumnDropperTransformer())]

    # Create percent change modifiers for the price and volume columns
    for c in ['Open','High','Low','Close','Volume']:
        layer_1.append((c+"pct",mods.PercentChangeTransformer(column=c,outputname="PctChange_" + c)))
        to_drop.append("PctChange_" + c)
        layer_2.append((c+"pct_der",mods.PercentChangeTransformer(column="PctChange_" + c,outputname="PctChange_" + c + "_der")))

    Percent_Change_FU = mods.DFFeatureUnion(transformer_list=layer_1,n_jobs=jobs)
    Percent_Change_Der_FU = mods.DFFeatureUnion(transformer_list=layer_2,n_jobs=jobs)
    bool_change_cols = []
    sma_layer_1 = [('pass',mods.ColumnDropperTransformer())]
    # Create SMA data for different periods
    for s in [4,8,16,32,64,128,20]:
        sma_layer_1.append(("SMA_" + str(s),mods.SMATransformer(column="Close",outputname='Period_%i_SMA'%s,period=s)))
        to_drop.append('Period_%i_SMA'%s)

    SMA_Close_Creation_FU = mods.DFFeatureUnion(transformer_list=sma_layer_1,n_jobs=jobs)

    sma_layer_2 = [('pass',mods.ColumnDropperTransformer()),('Close20Std',mods.STD2xTransformer(column='Close',outputname='Period_20_Close_Std',period=20))]
    BoolChangeLayer = [('pass',mods.ColumnDropperTransformer())]
    rip = []
    # For each of the SMA, determine what the crosses are
    for s in [4,8,16,20,32,64,128]:
        for d in [4,8,16,20,32,64,128]:
            sr = 'Period_%i_%i_GT'%(min(s,d),max(s,d))
            if(s == d or sr in rip):
                continue
            rip.append(sr)
            sma_layer_2.append((sr,
                mods.GreaterThanTransformer(
                columns=['Period_%i_SMA'%(min(s,d)),'Period_%i_SMA'%(max(s,d))]
                ,outputname=sr))
            )
            BoolChangeLayer.append(("cross_%i_%i"%(s,d),mods.BoolChangeTransformer(column=sr,outputname=sr+"_Cross")))

    SMA_GT_FU = mods.DFFeatureUnion(transformer_list=sma_layer_2,n_jobs=jobs)
    
    vol_layer_1 = [('pass',mods.ColumnDropperTransformer())]
    vol_layer_2 = [('pass',mods.ColumnDropperTransformer())]
    vol_layer_3 = [('pass',mods.ColumnDropperTransformer())]

    for s in [4,8,16,32]:
        vol_layer_1.append(("Period_" + str(s)+"_vol",mods.SMATransformer(column="Volume",outputname='Period_%i_Volume'%s,period=s)))
        to_drop.append('Period_%i_Volume'%s)
        vol_layer_2.append(("Period_" + str(s)+"_der",mods.PercentChangeTransformer(column='Period_%i_Volume'%s,outputname='Period_%i_Volume_Chg'%s)))
        vol_layer_3.append(("Period_" + str(s)+"_std",mods.STD2xTransformer(column='Volume',outputname='Period_%i_Volume_Std'%s,period=s)))
        to_drop.append('Period_%i_Volume_Std'%s)

    date_layer = [('pass',mods.ColumnDropperTransformer()),
                ('Month',mods.MonthTransformer()),
                ('Day',mods.DayTransformer()),
                ('Hour',mods.HourTransformer())]

    DateTime_Expand_FU = mods.DFFeatureUnion(transformer_list=date_layer,n_jobs=jobs)

    Volume_SMA_FU = mods.DFFeatureUnion(transformer_list=vol_layer_1,n_jobs=jobs)  
    Volume_PctChg_FU = mods.DFFeatureUnion(transformer_list=vol_layer_2,n_jobs=jobs) 
    Volume_STD_FU = mods.DFFeatureUnion(transformer_list=vol_layer_3,n_jobs=jobs) 

    rsi_layer_1 = [('pass',mods.ColumnDropperTransformer())]
    rsi_layer_2 = [('pass',mods.ColumnDropperTransformer())]
    for s in [7,14,30]:
        rsi_layer_1.append(('RSI_%i'%s,mods.RSITransformer(column='Close',outputname='Period_%i_Close_RSI'%s,period=s)))
        rsi_layer_2.append(('RSIChg_%i'%s,mods.PercentChangeTransformer(column='Period_%i_Close_RSI'%s,outputname='Period_%i_Close_RSI_PctChange'%s)))
    RSI_Creation_FU = mods.DFFeatureUnion(transformer_list=rsi_layer_1,n_jobs=jobs) 
    RSI_Change_FU = mods.DFFeatureUnion(transformer_list=rsi_layer_2,n_jobs=jobs) 


    bb_layer_1 = [('pass',mods.ColumnDropperTransformer())]
    bb_layer_2 =  [('pass',mods.ColumnDropperTransformer())]
    for s in [20]:
        bb_layer_1.append(("BBand_%i"%s,mods.BollingerBandTransform(smacolumn='Period_%i_SMA'%s,stdcolumn='Period_%i_Close_Std'%s,outputname="Period_%i_BBand"%s)))
        to_drop.append("Period_%i_BBand_Bot"%s)
        to_drop.append("Period_%i_BBand_Top"%s)
        for c in ['Open','Low','Close','High']:
            bb_layer_2.append( ("BBand_%s_GT_%i_B"%(c,s),mods.GreaterThanTransformer(columns=[c,"Period_%i_BBand_Bot"%s],outputname="Period_%i_BBand_Bot_GT_%s"%(s,c))))
            bb_layer_2.append( ("BBand_%s_GT_%i_T"%(c,s),mods.GreaterThanTransformer(columns=["Period_%i_BBand_Top"%s,c],outputname="Period_%i_BBand_Top_GT_%s"%(s,c))))

    BBand_Edges_FU = mods.DFFeatureUnion(transformer_list=bb_layer_1,n_jobs=jobs) 
    BBand_Edges_Cross_FU = mods.DFFeatureUnion(transformer_list=bb_layer_2,n_jobs=jobs) 
    ema_layer_1 = [('pass',mods.ColumnDropperTransformer())]
    for s in [12,26]:
        ema_layer_1.append(("ema_%s"%s,mods.EMATransformer(column="Close",outputname="Period_%i_Close_EMA"%s,period=s)))
        to_drop.append("Period_%i_Close_EMA"%s)
    Close_EMA_FU = mods.DFFeatureUnion(transformer_list=ema_layer_1,n_jobs=jobs)
    macd_layer_1 = [('pass',mods.ColumnDropperTransformer()),
                    ('macd',mods.MACDTransformer(macdcolumns=['Period_12_Close_EMA','Period_26_Close_EMA'],outputnames=['MACD_12_26_Close_Line','MACD_12_26_Close_Signal']))]
    macd_layer_2 = [('pass',mods.ColumnDropperTransformer())]
    Close_12_26_9_MACD = mods.DFFeatureUnion(transformer_list=macd_layer_1,n_jobs=jobs)
    macd_layer_2.append(("MACD_GT",mods.GreaterThanTransformer(columns=["MACD_12_26_Close_Line",'MACD_12_26_Close_Signal'],outputname="MACD_12_26_Close_Line_GT")))
    MACD_Cross_FU = mods.DFFeatureUnion(transformer_list=macd_layer_2,n_jobs=jobs)
    BoolChangeLayer.append(('MACD_12_26_Close_Line',mods.BoolChangeTransformer(column='MACD_12_26_Close_Line_GT',outputname="MACD_12_26_Close_Line_Cross")))
    to_drop.append('MACD_12_26_Close_Line')
    to_drop.append('MACD_12_26_Close_Signal')
    Cross_FU = mods.DFFeatureUnion(transformer_list=BoolChangeLayer,n_jobs=jobs)
    drops = mods.ColumnDropperTransformer(columns=to_drop)
    return Pipeline([
                    #('Percent Changes',Percent_Change_FU),
                    #('Percent Change Derivative',Percent_Change_Der_FU),
                    ('SMA',SMA_Close_Creation_FU),
                    ('SMA_Crosses',SMA_GT_FU),
                    
                    #('dates',DateTime_Expand_FU),
                    ('vol1',Volume_SMA_FU),
                    #('vol2',Volume_PctChg_FU),
                    ('volume_std',Volume_STD_FU),
                    ('rsi',RSI_Creation_FU),
                    #('rsi2',RSI_Change_FU),
                    ('bband',BBand_Edges_FU),
                    ('bband2',BBand_Edges_Cross_FU),
                    ('ema',Close_EMA_FU),
                    ('macd',Close_12_26_9_MACD),
                    ('macdcross',MACD_Cross_FU),
                    ('Crosses_Changed',Cross_FU),
                    ('drop',drops)
                    ])