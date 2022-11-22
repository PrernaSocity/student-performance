import matplotlib.pyplot as plt
import numpy as np
import random
import time
class eval1:
  x = []
  for i in range(200):
    x.append(i)
  y = [0.0024074917665622975, 0.006911509104639779, 0.010152335371678922, 0.024409419989778414, 0.03171498142627316, 0.03258800493150216, 0.036932614430297384, 0.038121077063990816, 0.04193210378545331, 0.045841636557705345, 0.05221841213638201, 0.05644422592312803, 0.06764872639006325, 0.06799969731086641, 0.07094005087824717, 0.0740942326936953, 0.07959685066076883, 0.082122284861179, 0.09050169962888843, 0.09099538357105996, 0.09543276080699192, 0.09584386935969103, 0.09638638504424568, 0.10388754603746941, 0.10856814848814378, 0.12636428077294615, 0.1276067393463286, 0.13680291853263682, 0.1415035353485492, 0.1597477014178048, 0.16049872016465339, 0.16082184742020855, 0.16100656320809947, 0.1623390496954341, 0.1746859892568916, 0.1767986263240925, 0.1774910084008472, 0.17905858422504484, 0.18122102474876367, 0.18659839621573004, 0.18720919863995555, 0.18723632593514172, 0.1900999586394393, 0.19688600227916864, 0.19846466821877629, 0.20129177613742755, 0.20475834012411587, 0.21363132993233291, 0.21843886721091021, 0.22526489944899808, 0.2261227696673055, 0.22910892739533606, 0.231562934163819, 0.23581240310537188, 0.24347472464753284, 0.24984392847866066, 0.25329190605801544, 0.26628210271923447, 0.27232722939995757, 0.2840373577079852, 0.2854294857312005, 0.2861826437750774, 0.30427290276153973, 0.3155063880712219, 0.32076491915759386, 0.32116384268643605, 0.3261064639416825, 0.3319322719358996, 0.35105362641567217, 0.35775390837726184, 0.375813161524547, 0.3803331341472881, 0.3817890129509436, 0.3821776595176676, 0.3856245326368042, 0.3870663279512472, 0.40782218607686993, 0.41443276896617753, 0.4227509094213545, 0.4248506455850053, 0.426311643817623, 0.43610817568401417, 0.44523311258006293, 0.4586874529096058, 0.4742314528306838, 0.4767761744318514, 0.47866358945273535, 0.4799125066040679, 0.4907498452314152, 0.4915624382211299, 0.4951520482467511, 0.4961689563287456, 0.5015906597604071, 0.5024191679638194, 0.5072260388389, 0.5103155199273469, 0.5138439605909094, 0.5142715114928701, 0.5208571035142043, 0.5278070595409595, 0.5280608401150974, 0.5296601936420404, 0.5339610049030145, 0.5377081602812995, 0.5383779511146237, 0.5394469613505201, 0.539507516792253, 0.5399497907500482, 0.544801953095578, 0.5490282660081405, 0.5521139878700494, 0.5658751534073525, 0.5665340026475059, 0.5694418766784503, 0.5715385612019315, 0.5777070548125233, 0.5782575571982365, 0.5797223103524439, 0.590823675572344, 0.6141659219172693, 0.618490164356103, 0.6188758523553837, 0.6221609760996235, 0.6299033375994791, 0.6306617289622447, 0.6334996931496648, 0.6346579079230332, 0.6373207072767257, 0.6440771960298745, 0.651826928538971, 0.6682636261028313, 0.6818059115294317, 0.6834594833371872, 0.6952188419448431, 0.6980993228331804, 0.6992717136009878, 0.7093122839398237, 0.7145371672575104, 0.7194278522330121, 0.7208450735597349, 0.7229553989301598, 0.7241097561000626, 0.7249591605607609, 0.7267761994606672, 0.7342991328613813, 0.7413175633765369, 0.7531022063514073, 0.7547039707751403, 0.761360806333522, 0.7620617450380808, 0.768307995091614, 0.7747551366015595, 0.7754594205461516, 0.7806383593109048, 0.7848692472203503, 0.7929734424083871, 0.7972206230721394, 0.8048328967466426, 0.8071721474293566, 0.8119680201930967, 0.813815920886904, 0.814764958049872, 0.8171438281634922, 0.8194442963275602, 0.8223355433253628, 0.823434079642767, 0.8301823515255988, 0.8366733566158611, 0.8431510623607263, 0.8547644502842908, 0.8573120599142064, 0.8609630278683028, 0.862107099864716, 0.8652577778968398, 0.8708467459099415, 0.8711440926620245, 0.8811043402635103, 0.8841535200770753, 0.8918876621050195, 0.8953788847930138, 0.8995013425127468, 0.9019053191267368, 0.9082172203828999, 0.9126630233363662, 0.9175061497507194, 0.9180188741990477, 0.9239828670575194, 0.9286566605152353, 0.9308998559642689, 0.9367583690097412, 0.942845708413537, 0.9436780670286185, 0.9493564270652123, 0.95363305206359, 0.9674629722537197, 0.968525780411927, 0.974243470236296, 0.9789341564598515, 0.9939834113176034, 0.9959307191295201]
  y1 = [0.00954420137580081, 0.009572357792675845, 0.009572357792675845, 0.009572357792675845, 0.009572357792675845, 0.025348718268649373, 0.03609405757254103, 0.03851678968266303, 0.03895238107172394, 0.047102280391785434, 0.04748089465938943, 0.04917919676231852, 0.049444834296306306, 0.05351397905389377, 0.0656625943006588, 0.07374667725077078, 0.09303772530675924, 0.0957496680293708, 0.09637785756420858, 0.09812026527850504, 0.09860054407384011, 0.10262381024192913, 0.1027859666625105, 0.10699646586060185, 0.10710666561889892, 0.10730790380001287, 0.10827023411704362, 0.11478658055983348, 0.11600341830304273, 0.12098901753172764, 0.12486517554644405, 0.1258440434527781, 0.13323755849047092, 0.13779004044278942, 0.14021950535911132, 0.15044827554682183, 0.15855388227359157, 0.16230256901790452, 0.17115179479589326, 0.17171719175806965, 0.17493613174281297, 0.17677214861393298, 0.17885060817826925, 0.19156520521555687, 0.20344663489470827, 0.2070637974364461, 0.21198911498421424, 0.2129857902697908, 0.23263805305233776, 0.23675287240811882, 0.24289872006502933, 0.2438835209074891, 0.24891440537486575, 0.25268094894367377, 0.2568770431893439, 0.26049431811958357, 0.2653207456757912, 0.26938035284871253, 0.2737188221171172, 0.2984897896332548, 0.3008005944513751, 0.3130921487722882, 0.3238342358006009, 0.32557324223966544, 0.3303737631497462, 0.3357709261075805, 0.3437240677066633, 0.34965296685791325, 0.3518231609528981, 0.3525144846945393, 0.35923408051869765, 0.3732653336495897, 0.3771177661843531, 0.37918191807362855, 0.38171565139206587, 0.39024161402436874, 0.3998201778945114, 0.40144501957136824, 0.40304380029013076, 0.40434237738276035, 0.40483598416010147, 0.40881423462224464, 0.40928641610025374, 0.4101691616867186, 0.41231819850687756, 0.41715433735671303, 0.42042426667134447, 0.43504843670083215, 0.4355149358587256, 0.43837730530324204, 0.43890975102985386, 0.4446738396294285, 0.447011963230459, 0.4516281450912356, 0.46993744700645945, 0.4717185726652331, 0.47400989949231553, 0.4767208048472543, 0.4775026489654619, 0.4781709471064788, 0.47928120256870044, 0.4871931193114043, 0.49178219952606184, 0.5004764024178053, 0.5018267704087076, 0.5056498567792073, 0.5060009402862318, 0.50647161700618, 0.5149442182029428, 0.520623013054189, 0.535217451661813, 0.5517032157407038, 0.5543962943849258, 0.5545294688938921, 0.6009360469107367, 0.6010893346714606, 0.6019788063205553, 0.6077635248514396, 0.6102400023100646, 0.6139311457798542, 0.6179114918909425, 0.6247109342448097, 0.6277395355102016, 0.6284693948903967, 0.6305694032341961, 0.6328132494451264, 0.6368109845253263, 0.637265443033646, 0.643752326225508, 0.6465237730252522, 0.6517305421747815, 0.6527244597084413, 0.6668109894100643, 0.6931706543100321, 0.6941434523806161, 0.6953160950676301, 0.6977586444355877, 0.698505483771283, 0.7002584317045822, 0.7058169011808313, 0.7092920895191609, 0.7186723494282867, 0.7196131832630853, 0.7374410987464706, 0.741250232735933, 0.7519306540608953, 0.7544344242411787, 0.756753695745889, 0.7599812968241099, 0.761680827361369, 0.7700456113985661, 0.7790046225275519, 0.7803350420227777, 0.7807363987864576, 0.7824427678884919, 0.7880972542949024, 0.7883149829607716, 0.8040375776618572, 0.804553823708974, 0.8118778910502751, 0.8118991766203915, 0.8122875076086599, 0.8144895705196434, 0.8161746502202999, 0.8207693558124116, 0.8240121368027907, 0.8283687967847538, 0.8327457748697658, 0.8389664022420491, 0.8420306799010541, 0.8513299049657246, 0.8573723697735648, 0.8679192403975285, 0.8750573729289274, 0.8828600849390923, 0.8847901784658949, 0.8968874300040621, 0.9056841197098456, 0.918465557887873, 0.9191801755890444, 0.9216955220929937, 0.9233645373757634, 0.9235096088700248, 0.925672030013636, 0.9284983938523507, 0.9301181878541689, 0.9428056803815794, 0.9466477766520861, 0.9491779617620927, 0.9535809888797294, 0.9537670257516918, 0.9628773774502278, 0.9700878446769186, 0.9424138620997174, 0.9424894974184258, 0.9479013022309834, 0.9448706804791996, 0.9474438160116692, 0.9447657598130443, 0.9452353005739305]
  for i in y1[25:76]:
    y1[y1.index(i)] = 0.025348718268649373
  for i in y1[175:200]:
    y1[y1.index(i)] = 0.9452353005739305
  for i in y1[125:150]:
    y1[y1.index(i)] = 0.7883149829607716
  for i in y[25:76]:
    y[y.index(i)] = 0.035348718268649373
  for i in y[175:200]:
    y[y.index(i)] = 0.9552353005739305
  for i in y[125:150]:
    y[y.index(i)] = 0.7983149829607716
  x12 = time.sleep(3600)