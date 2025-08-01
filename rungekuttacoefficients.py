import mpmath as mp
import numpy as np
from mpmath import mpmathify
#order 45 coefficients
def rk45Ak():
    return mp.matrix(
        [
            mpmathify(0),
            mpmathify(1) / mpmathify(4),
            mpmathify(3) / mpmathify(8),
            mpmathify(12) / mpmathify(13),
            mpmathify(1),
            mpmathify(1) / mpmathify(2),
        ]
    )

def rk45Ck():
    return mp.matrix(
        [
            mpmathify(25) / mpmathify(216),
            mpmathify(0),
            mpmathify(1408) / mpmathify(2565),
            mpmathify(2197) / mpmathify(4104),
            mpmathify(-1) / mpmathify(5),
            mpmathify(0),
        ]
    )
def rk45CHk():
    return mp.matrix(
        [
            mpmathify(16) / mpmathify(135),
            mpmathify(0),
            mpmathify(6656) / mpmathify(12825),
            mpmathify(28561) / mpmathify(56430),
            mpmathify(-9) / mpmathify(50),
            mpmathify(2) / mpmathify(55),
        ]
    )

def rk45CTk():
    return mp.matrix(
        [
            mpmathify(-1) / mpmathify(360),
            mpmathify(0),
            mpmathify(128) / mpmathify(4275),
            mpmathify(2197) / mpmathify(75240),
            mpmathify(-1) / mpmathify(50),
            mpmathify(-2) / mpmathify(55),
        ]
    )
def rk45Bkl():
    return mp.matrix(
        [
            [
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(1) / mpmathify(4),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(3) / mpmathify(32),
                mpmathify(9) / mpmathify(32),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(1932) / mpmathify(2197),
                mpmathify(-7200) / mpmathify(2197),
                mpmathify(7296) / mpmathify(2197),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(439) / mpmathify(216),
                mpmathify(8) / mpmathify(-1),
                mpmathify(3680) / mpmathify(513),
                mpmathify(-845) / mpmathify(4104),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(-8) / mpmathify(27),
                mpmathify(2) / mpmathify(1),
                mpmathify(-3544) / mpmathify(2565),
                mpmathify(1859) / mpmathify(4104),
                mpmathify(-11) / mpmathify(40),
                mpmathify(0),
            ],
        ]
    )

#Hgher order 89 coefficients

def rk89CHk():
    return mp.matrix(
        [
            mpmathify("0.32256083500216249913612900960247e-1"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.25983725283715403018887023171963"),
            mpmathify("0.92847805996577027788063714302190e-1"),
            mpmathify("0.16452339514764342891647731842800"),
            mpmathify("0.17665951637860074367084298397547"),
            mpmathify("0.23920102320352759374108933320941"),
            mpmathify("0.39484274604202853746752118829325e-2"),
            mpmathify("0.30726495475860640406368305522124e-1"),
            mpmathify("0.0"),
            mpmathify("0.0"),
   
        ]
    )


def rk89Ak():
    return mp.matrix(
        [
            mpmathify(" 0.0"),
        mpmathify(" 0.44368940376498183109599404281370"),
        mpmathify(" 0.66553410564747274664399106422055"),
        mpmathify(" 0.99830115847120911996598659633083"),
        mpmathify(" 0.3155"),
        mpmathify(" 0.50544100948169068626516126737384"),
        mpmathify(" 0.17142857142857142857142857142857"),
        mpmathify(" 0.82857142857142857142857142857143"),
        mpmathify(" 0.66543966121011562534953769255586"),
        mpmathify(" 0.24878317968062652069722274560771"),
        mpmathify(" 0.1090"),
        mpmathify(" 0.8910"),
        mpmathify(" 0.3995"),
        mpmathify(" 0.6005"),
        mpmathify(" 1.0"),
        mpmathify("0.0"),
        mpmathify(" 1.0"),
        ]
    )

def rk89Bkl():
    return mp.matrix(
        [
[mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify(" 0.44368940376498183109599404281370"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.16638352641186818666099776605514"),
mpmathify("  0.49915057923560455998299329816541"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.24957528961780227999149664908271"),
mpmathify(" 0.0"),
mpmathify("  0.74872586885340683997448994724812"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.20661891163400602426556710393185"),
mpmathify(" 0.0"),
mpmathify("  0.17707880377986347040380997288319"),
mpmathify("  -0.68197715413869494669377076815048e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.10927823152666408227903890926157"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.40215962642367995421990563690087e-2"),
mpmathify("  0.39214118169078980444392330174325"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.98899281409164665304844765434355e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.35138370227963966951204487356703e-2"),
mpmathify("  0.12476099983160016621520625872489"),
mpmathify("  -0.55745546834989799643742901466348e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.36806865286242203724153101080691"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.22273897469476007645024020944166e+1"),
mpmathify("  0.13742908256702910729565691245744e+1"),
mpmathify("  0.20497390027111603002159354092206e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.45467962641347150077351950603349e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.32542131701589147114677469648853"),
mpmathify("  0.28476660138527908888182420573687"),
mpmathify("  0.97837801675979152435868397271099e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.60842071062622057051094145205182e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.21184565744037007526325275251206e-1"),
mpmathify("  0.19596557266170831957464490662983"),
mpmathify("  -0.42742640364817603675144835342899e-2"),
mpmathify("  0.17434365736814911965323452558189e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.54059783296931917365785724111182e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.11029825597828926530283127648228"),
mpmathify("  -0.12565008520072556414147763782250e-2"),
mpmathify("  0.36790043477581460136384043566339e-2"),
mpmathify("  -0.57780542770972073040840628571866e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.12732477068667114646645181799160"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.11448805006396105323658875721817"),
mpmathify("  0.28773020709697992776202201849198"),
mpmathify("  0.50945379459611363153735885079465"),
mpmathify("  -0.14799682244372575900242144449640"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.36526793876616740535848544394333e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.81629896012318919777819421247030e-1"),
mpmathify("  -0.38607735635693506490517694343215"),
mpmathify("  0.30862242924605106450474166025206e-1"),
mpmathify("  -0.58077254528320602815829374733518e-1"),
mpmathify("  0.33598659328884971493143451362322"),
mpmathify("  0.41066880401949958613549622786417"),
mpmathify("  -0.11840245972355985520633156154536e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("   -0.12375357921245143254979096135669e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("   -0.24430768551354785358734861366763e+2"),
mpmathify("  0.54779568932778656050436528991173"),
mpmathify("   -0.44413863533413246374959896569346e+1"),
mpmathify("  0.10013104813713266094792617851022e+2"),
mpmathify("   -0.14995773102051758447170985073142e+2"),
mpmathify("  0.58946948523217013620824539651427e+1"),
mpmathify("  0.17380377503428984877616857440542e+1"),
mpmathify("  0.27512330693166730263758622860276e+2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.35260859388334522700502958875588"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.18396103144848270375044198988231"),
mpmathify("  -0.65570189449741645138006879985251"),
mpmathify("  -0.39086144880439863435025520241310"),
mpmathify("  0.26794646712850022936584423271209"),
mpmathify("  -0.10383022991382490865769858507427e+1"),
mpmathify("  0.16672327324258671664727346168501e+1"),
mpmathify("  0.49551925855315977067732967071441"),
mpmathify("  0.11394001132397063228586738141784e+1"),
mpmathify("  0.51336696424658613688199097191534e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.10464847340614810391873002406755e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.67163886844990282237778446178020e-2"),
mpmathify("  0.81828762189425021265330065248999e-2"),
mpmathify("  -0.42640342864483347277142138087561e-2"),
mpmathify("  0.28009029474168936545976331153703e-3"),
mpmathify("  -0.87835333876238676639057813145633e-2"),
mpmathify("  0.10254505110825558084217769664009e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.13536550786174067080442168889966e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.18396103144848270375044198988231"),
mpmathify("  -0.65570189449741645138006879985251"),
mpmathify("  -0.39086144880439863435025520241310"),
mpmathify("  0.27466285581299925758962207732989"),
mpmathify("  -0.10464851753571915887035188572676e+1"),
mpmathify("  0.16714967667123155012004488306588e+1"),
mpmathify("  0.49523916825841808131186990740287"),
mpmathify("  0.11481836466273301905225795954930e+1"),
mpmathify("  0.41082191313833055603981327527525e-1"),
mpmathify(" 0.0"),
mpmathify("   1.0"),
mpmathify(" 0.0"),
],
]
    )