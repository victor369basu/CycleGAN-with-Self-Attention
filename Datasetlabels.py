import numpy as np
## glass
domainG = list(['face-1001.png','face-1052.png','face-1064.png','face-1012.png',
              'face-1029.png','face-1126.png','face-1023.png','face-1130.png',
              'face-1124.png','face-1122.png','face-1061.png','face-1069.png',
              'face-1139.png','face-1208.png','face-1166.png','face-10.png',
              'face-1068.png','face-1011.png','face-1053.png'
              ])
## noglass
domainNG = list(['face-1.png','face-1007.png','face-1021.png','face-1060.png', 
                'face-1078.png','face-1157.png', 'face-1148.png', 'face-1176.png',
                'face-1026.png','face-1032.png','face-1079.png','face-11.png',
                'face-1125.png','face-1170.png','face-1173.png','face-1227.png',
                'face-1041.png','face-1027.png','face-1032.png'])
## boys
domainB=list(['face-1.png','face-1004.png','face-1160.png','face-1022.png',
      'face-1025.png','face-1027.png','face-1028.png','face-1032.png',
      'face-1044.png','face-1010.png','face-1069.png','face-1078.png',
      'face-1141.png','face-1227.png','face-1304.png','face-1319.png',
      'face-1326.png','face-1379.png','face-1471.png','face-1799.png'
      ])
## girls
domainG=list(['face-1007.png','face-1009.png','face-1024.png','face-1021.png',
       'face-105.png','face-1050.png','face-1057.png','face-1079.png',
       'face-1101.png','face-1191.png','face-1267.png','face-1295.png',
       'face-1318.png','face-1378.png','face-1455.png','face-1509.png',
       'face-151.png','face-1560.png','face-174.png','face-1792.png'])
train_imgs = np.array(domainG + domainNG)