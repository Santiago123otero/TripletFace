Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

![TSNE_Latent](TSNE_Latent.png)

## Architecture

The proposed architecture is pretty simple and does not implement state of the
art performances. The chosen architecture is a fine tuning example of the
resnet18 CNN model. The model includes the freezed CNN part of resnet, and its
FC part has been replaced to be trained to output latent variables for the
facial image input.

The dataset needs to be formatted in the following form:
```
dataset/
| test/
| | 0/
| | | 00563.png
| | | 01567.png
| | | ...
| | 1/
| | | 00011.png
| | | 00153.png
| | | ...
| | ...
| train/
| | 0/
| | | 00001.png
| | | 00002.png
| | | ...
| | 1/
| | | 00001.png
| | | 00002.png
| | | ...
| | ...
| labels.csv        # id;label
```

## Install

Install all dependencies ( pip command may need sudo ):
```bash
cd TripletFace/
pip3 install -r requirements.txt
```

## Usage

For training:
```bash
usage: train.py [-h] -s DATASET_PATH -m MODEL_PATH [-i INPUT_SIZE]
                [-z LATENT_SIZE] [-b BATCH_SIZE] [-e EPOCHS]
                [-l LEARNING_RATE] [-w N_WORKERS] [-r N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s DATASET_PATH, --dataset_path DATASET_PATH
  -m MODEL_PATH, --model_path MODEL_PATH
  -i INPUT_SIZE, --input_size INPUT_SIZE
  -z LATENT_SIZE, --latent_size LATENT_SIZE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w N_WORKERS, --n_workers N_WORKERS
  -r N_SAMPLES, --n_samples N_SAMPLES
```

## References

* Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
* Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
* TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

**In the file = Project1.ipynb i showed my work in google colab.**

The students are asked to complete the following tasks:
* Fork the Project: **check**
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet ): **Check**

```bash
parser        = argparse.ArgumentParser( )
parser.add_argument( '-s', '--dataset_path',  type = str,   required = True )
parser.add_argument( '-m', '--model_path',    type = str,   required = True )
parser.add_argument( '-i', '--input_size',    type = int,   default  = 224 )
parser.add_argument( '-z', '--latent_size',   type = int,   default  = 32 )
parser.add_argument( '-b', '--batch_size',    type = int,   default  = 16 )
parser.add_argument( '-e', '--epochs',        type = int,   default  = 10 )
parser.add_argument( '-l', '--learning_rate', type = float, default  = 1e-3 )
parser.add_argument( '-w', '--n_workers',     type = int,   default  = 4 )
parser.add_argument( '-r', '--n_samples',     type = int,   default  = 6 )
```
![TSNE_Latent](vizualisation_5.png)

* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )

**i was able to generate the jitcompile.pt successfully but the file is too big to uploaded as well as my model.pt** 

```bash
import torch

from tripletface.core.model import Encoder

model = Encoder(64)
weights = torch.load( "model/model.pt" )['model']
model.load_state_dict( weights )
jit_model = torch.jit.trace( model, torch.rand(4, 3, 4, 3) )
torch.jit.save( jit_model, "model/jitcompile.pt" )
```

* Add script to generate Centroids and Thesholds using few face images from one person: 
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model

**I used this script that was not made by me. i get the centroids and the threshelods kind of but i am know this is not what you are asking, however i posted the results because i think they are very interesting.
(triplet_inference.py)**

* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )

People 0: 
Tresholds = 0.40548295181128197
Centroid = [-0.14217476057820022, -0.022368336794897914, 0.16067822329932824, 0.50144973577698693, -0.48882991401478648, -0.077571916626766324, -0.17792163797275862, 0.14542136644013226, -0.56713610514998436, 0.5711914780549705, -0.84808010421693325, 0.10770503955427557, -0.10127248871140182, 0.87072942405939102, 0.25426830386277288, -0.50021783215925097, -0.63492217950988561, -0.58845732873305678, -0.60083950031548738, 1.2519282605499029, 0.034675421076826751, 0.21098475164035335, 0.6738878795877099, 0.070157515758182853, 0.19918441656045616, 0.13983295917569194, -0.040799407695885748, -0.39680884953122586, -0.074149309424683452, -0.76377893425524235, 0.31727452366612852, -0.24477353412657976, -0.13016357336891815, 0.19469253357965499, 0.20853172987699509, -0.20956118276808411, -0.96204543113708496, -0.60385395772755146, 0.86780743580311537, -0.38821523962542415, 0.0016436458099633455, -0.10197210044134408, 0.21503199287690222, 0.43207545927725732, -0.82517835032194853, 0.028893173919641413, -0.51204833062365651, -0.02128394867759198, -0.24959297344321385, -0.35138676490169019, -0.48648670560214669, 0.58455936703830957, -0.31719820667058229, -0.046314158360473812, 0.81319827493280172, 0.14207125024404377, -0.398909917101264, -0.33589987561572343, -0.34557322692126036, 0.45300947327632457, 0.10349104704800993, -0.056229433510452509, 0.16533716069534421, 0.35803522437345237]

People 1: 
Tresholds = 0.44406544548110105
Centroid = [-0.10303625743836164, -0.045986008830368519, 0.1498136674053967, 0.4801123776123859, -0.43576056614983827, -0.058341167867183685, -0.17673155339434743, 0.12908784177852795, -0.57900692359544337, 0.56641570362262428, -0.83705627406015992, 0.14405282004736364, -0.072180037386715412, 0.8743248051032424, 0.24567387229762971, -0.49119351687841117, -0.61133973544929177, -0.59423406422138214, -0.60561236459761858, 1.1894417814910412, -0.0081475525512360036, 0.22124140377854928, 0.62388075795024633, 0.079050684755202383, 0.25298564566764981, 0.092209757101954892, -0.023222200252348557, -0.39627430390100926, -0.071878907270729542, -0.75020069070160389, 0.28169048301060684, -0.28859122470021248, -0.10039281961508095, 0.15742060949560255, 0.20523828669684008, -0.15925760730169713, -0.95922043081372976, -0.58800115183839807, 0.82158963941037655, -0.40161927428562194, -0.0060414084000512958, -0.097490482614375651, 0.22172447212506086, 0.41869884287007153, -0.78798563592135906, 0.0041749753872863948, -0.52102333481889218, -0.035000185249373317, -0.23248717439128086, -0.33508887962670997, -0.45838534773793072, 0.56641141045838594, -0.28226901637390256, -0.039852516376413405, 0.76527436077594757, 0.13182861276436597, -0.42910206597298384, -0.29538452904671431, -0.36478802608326077, 0.42487521725706756, 0.066246225993381813, -0.089425958693027496, 0.1343063972890377, 0.30842950800433755]

People 2: 
Tresholds = 0.4471329806582071
Centroid = [-0.11898765014484525, -0.036233294871635735, 0.1072742035612464, 0.49966987146763131, -0.44181284005753696, -0.076633184216916561, -0.16017391931382008, 0.12757519434671849, -0.56949062296189368, 0.56607535295188427, -0.83152974164113402, 0.17668567830696702, -0.11067784321494401, 0.85378566244617105, 0.23220238753128797, -0.48466359660960734, -0.61445546487811953, -0.56183788413181901, -0.63255092827603221, 1.2143191564828157, 0.05975204921560362, 0.22460264223627746, 0.66110390517860651, 0.080793495406396687, 0.24056784913409501, 0.14746925969666336, -0.027673322358168662, -0.41414832998998463, -0.061036018887534738, -0.77909712959080935, 0.26939990189566743, -0.31193965941201895, -0.15400437079370022, 0.19678606709931046, 0.19988131066202186, -0.17227068496868014, -0.95076623372733593, -0.59673829446546733, 0.83502767072059214, -0.39433403732255101, -0.0040253550978377461, -0.13019590417388827, 0.20190241129603237, 0.38750270602758974, -0.81266470160335302, -0.00036868758616037667, -0.52356185624375939, -0.063141950173303485, -0.25154803466284648, -0.37524222805222962, -0.43187218764796853, 0.56273575080558658, -0.25009179348126054, -0.044143376406282187, 0.80901812948286533, 0.13885124074295163, -0.44484919123351574, -0.26653798727784306, -0.34166949149221182, 0.41739773083827458, 0.094047531485557556, -0.10086609842255712, 0.14091135002672672, 0.30191425723023713]

People 3: 
Tresholds = 0.3878582546807593
Centroid = [-0.13044312549754977, 0.0058577731251716614, 0.17249781364807859, 0.51085186860291287, -0.44898017682135105, -0.070040161954239011, -0.19772514421492815, 0.15649837011005729, -0.58530439483001828, 0.56769903586246073, -0.82878314983099699, 0.13017981010489166, -0.10263234458398074, 0.89671700168401003, 0.24450975668150932, -0.46527073113247752, -0.65467849385458976, -0.62073180498555303, -0.59894770022947341, 1.2597004976123571, 0.034407100465614349, 0.23381185060134158, 0.68604911724105477, 0.11554173234617338, 0.20818395348032936, 0.12482648601871915, -0.0079575395211577415, -0.40610396978445351, -0.045649873558431864, -0.74455029936507344, 0.27897641813979135, -0.28548899618908763, -0.16327042059856467, 0.20003684773109853, 0.22323264888837002, -0.17168226954527199, -0.94767717737704515, -0.61239198502153158, 0.86960329208523035, -0.39027217100374401, -0.008062398643232882, -0.057167986989952624, 0.187696099630557, 0.41521550016477704, -0.87432573456317186, -0.019123268124531023, -0.52028901723679155, 0.0036264962982386351, -0.26573812705464661, -0.38037669083860237, -0.42129041242878884, 0.60250696097500622, -0.28293826896697283, -0.038385594496503472, 0.81233678502030671, 0.12478851189371198, -0.41190420743077993, -0.31521560973487794, -0.34545637154951692, 0.45660353067796677, 0.053713352594058961, -0.080866544041782618, 0.16710983635857701, 0.33282962220255286]

People 4: 
Tresholds = 0.3882316666073166
Centroid = [-0.096210963791236281, 0.00035351235419511795, 0.14416584483115003, 0.49390878627309576, -0.43805750110186636, -0.05358912143856287, -0.20692008296464337, 0.17344712279736996, -0.57733920565806329, 0.5483391263987869, -0.82801920548081398, 0.12279922678135335, -0.1049151869956404, 0.90125533193349838, 0.22073160973377526, -0.45573131367564201, -0.63081052992492914, -0.5979824117384851, -0.59342207573354244, 1.2458915747702122, 0.013278815429657698, 0.23693371383706108, 0.67025787895545363, 0.10997349658282474, 0.20899951690807939, 0.10085270903073251, -0.036140465410426259, -0.41025344375520945, -0.067414693534374237, -0.75698802387341857, 0.28579372842796147, -0.27926773822400719, -0.18846530665177852, 0.20507696305867285, 0.20702476854785345, -0.17020453896839172, -0.9469109084457159, -0.59926971657114336, 0.87680183723568916, -0.39679206727305427, 0.0061980277532711625, -0.076609838055446744, 0.20477227319497615, 0.40869929525069892, -0.84186153579503298, 0.00064383215794805437, -0.51748892420437187, -0.010520651587285101, -0.24748859100509435, -0.35012410837225616, -0.4075480995234102, 0.64259157981723547, -0.2672684354474768, -0.037223018007352948, 0.79265734367072582, 0.12433232995681465, -0.42463725060224533, -0.31738600262906402, -0.39178826101124287, 0.43215750064700842, 0.077798939950298518, -0.063291739439591765, 0.15519466297701001, 0.35056169133167714]

People 5: 
Tresholds = 0.4291277751876623
Centroid = [-0.12110535800457001, -0.027109497110359371, 0.14413086674176157, 0.50385182729223743, -0.48196094855666161, -0.06320977327413857, -0.17960896156728268, 0.13273336680140346, -0.58652717410586774, 0.56208848650567234, -0.84561555413529277, 0.12616016005631536, -0.12066867738030851, 0.87877126503735781, 0.25026403914671391, -0.47826009779237211, -0.63138359412550926, -0.59322927758330479, -0.60451137647032738, 1.2481417302042246, 0.012449128087610006, 0.22434064926346764, 0.65991542348638177, 0.10046082775807008, 0.23300057952292264, 0.13102578955476929, -0.056795568612869829, -0.42752681521233171, -0.087900442769750953, -0.76194129232317209, 0.29221975851760362, -0.26122440141625702, -0.15766026498749852, 0.19140452169813216, 0.2146515172207728, -0.20203486341051757, -0.97157491929829121, -0.60217211465351284, 0.86501906649209559, -0.38184052245924249, -0.016937328735366464, -0.093606295878998935, 0.24146988580469042, 0.42696788895409554, -0.87240061536431313, 0.010104654735187069, -0.52095563849434257, 0.0062474862206727266, -0.2695771285216324, -0.35798501851968467, -0.43384049239102751, 0.57984611857682467, -0.27001101628411561, -0.031796674244105816, 0.82328569982200861, 0.12512196419993415, -0.43166749086230993, -0.33330912655219436, -0.37107422755798325, 0.4692740423779469, 0.088091848912881687, -0.061245059361681342, 0.15799379046075046, 0.32718662964180112]

People 6: 
Tresholds = 0.3998356171738124
Centroid = [-0.10975669836625457, -0.0017972987843677402, 0.15520004002610222, 0.50041811540722847, -0.46843084949068725, -0.071873667882755399, -0.19399713375605643, 0.14588042336981744, -0.5806726801674813, 0.57827873155474663, -0.83556356653571129, 0.13911195774562657, -0.086584782460704446, 0.86795326322317123, 0.25915276014711708, -0.46939195040613413, -0.61757465580012649, -0.61058233864605427, -0.60487148011452518, 1.2420619409531355, -0.0049112407141365111, 0.21562457602703944, 0.67663575755432248, 0.095753678819164634, 0.21744545421097428, 0.13187389393351623, -0.027882211899850518, -0.42224815941881388, -0.088432351127266884, -0.77870513219386339, 0.31387730338610709, -0.29561588005162776, -0.14431316845002584, 0.18428086151834577, 0.19941890391055495, -0.1919840598711744, -0.92800892516970634, -0.59943335549905896, 0.8537011481821537, -0.35955275117885321, 0.021000737207941711, -0.10035865963436663, 0.21065424929838628, 0.41585046052932739, -0.83638696488924325, 0.004559673776384443, -0.52692406496498734, -0.023631804506294429, -0.22671849559992552, -0.36598171980585903, -0.40342415554914623, 0.56229962012730539, -0.26079120545182377, -0.049835620215162635, 0.81797924131387845, 0.13610620843246579, -0.41799296159297228, -0.35492438595974818, -0.35173005430260673, 0.44308541365899146, 0.081560093676671386, -0.041082462179474533, 0.1538934470154345, 0.32933633029460907]

People 7: 
Tresholds = 0.3897659898304846
Centroid = [-0.11542332777753472, -0.023269796860404313, 0.15940456045791507, 0.51743648254341679, -0.45438851334620267, -0.062293519265949726, -0.17674514232203364, 0.12366234138607979, -0.58767980011180043, 0.56827090249862522, -0.81638410873711109, 0.1298697660677135, -0.11347575369291008, 0.85947789344936609, 0.27176036429591477, -0.44580676732584834, -0.63217389956116676, -0.5913471756502986, -0.61593067133799195, 1.2243341468274593, 0.01449056196724996, 0.23700609937077388, 0.65761497151106596, 0.08750953315757215, 0.23401876364368945, 0.12755141349043697, -0.033203059458173811, -0.37793685216456652, -0.080975242890417576, -0.77627294836565852, 0.28912725072586909, -0.27509226987604052, -0.12885421814280562, 0.16925510915461928, 0.20370281953364611, -0.1689894306473434, -0.94092445075511932, -0.56655101254000328, 0.8671602567192167, -0.38021133103757165, 0.0057163925375789404, -0.089597088168375194, 0.21617248450638726, 0.43519171338994056, -0.83701065182685852, 0.0494158681249246, -0.51163167413324118, -0.027379573788493872, -0.23327308357693255, -0.34169429511530325, -0.44253083958756179, 0.56417644070461392, -0.29371122911106795, -0.032735558575950563, 0.81970404088497162, 0.11419303994625807, -0.44138970784842968, -0.30973943613935262, -0.37836793757742271, 0.43766616654465906, 0.075281868601450697, -0.068031210452318192, 0.13056253967806697, 0.34525034960824996]

People 8: 
Tresholds = 0.3764893719542306
Centroid = [-0.13111365353688598, -0.054978119907900691, 0.16279217053670436, 0.49584314087405801, -0.45605083485133946, -0.047235438600182533, -0.17114771508931881, 0.13079552893759683, -0.58818652550689876, 0.56528023304417729, -0.83481089677661657, 0.13100444921292365, -0.096252641174942255, 0.91374163143336773, 0.25737437780480832, -0.45574659993872046, -0.6053062817081809, -0.58359724609181285, -0.61372657772153616, 1.224616979714483, -0.013049904373474419, 0.22005408891709521, 0.63326062355190516, 0.098179611552041024, 0.22242023167200387, 0.13951786985853687, -0.0051358510972931981, -0.37999359262175858, -0.05129266669973731, -0.77858881093561649, 0.30763128404214513, -0.25291772291529924, -0.14115015597781166, 0.15041362051852047, 0.22043392175692134, -0.17298630881123245, -0.93329527974128723, -0.58804679641616531, 0.88433868810534477, -0.36085234431084245, 0.0085694547742605209, -0.072087408974766731, 0.20441987673984841, 0.39586187631357461, -0.82362764049321413, 0.0065276759123662487, -0.50647401309106499, -0.043408469995483756, -0.20801560021936893, -0.34958552266471088, -0.42857355216983706, 0.58099043625406921, -0.28660158824641258, -0.02249395614489913, 0.82235172344371676, 0.11896383354905993, -0.43135849386453629, -0.30345799005590379, -0.36131986550753936, 0.43403223308268934, 0.070420356903923675, -0.047193419653922319, 0.13744078343734145, 0.30932420922908932]

People 9: 
Tresholds = 0.38866480064345527
Centroid = [-0.11819981527514756, -0.0069071851903572679, 0.17454045847989619, 0.50888428563484922, -0.49506036646198481, -0.059675928670912981, -0.18336876158718951, 0.14297599520068616, -0.58776834490709007, 0.57648044999223202, -0.83508439781144261, 0.11002058687154204, -0.097527498146519065, 0.86714540654793382, 0.29409389721695334, -0.46966587565839291, -0.65516326623037457, -0.62487266305834055, -0.62446677929256111, 1.2869849670678377, 0.049099729629233479, 0.24362997530261055, 0.71950610494241118, 0.074510985519737005, 0.21498029807116836, 0.1333315692609176, 0.0025998084747698158, -0.418141522211954, -0.089958568336442113, -0.7655621599406004, 0.30014190054498613, -0.28639101516455412, -0.12900282995542511, 0.21068706980440766, 0.23654383289976977, -0.23704207304399461, -0.94239278312306851, -0.60243649783660658, 0.83839613012969494, -0.41970792025676928, -0.01174110546708107, -0.075539315468631685, 0.22088031703606248, 0.42641075036954135, -0.87618972454220057, -0.0086087366653373465, -0.51237272366415709, -0.0054955229861661792, -0.27135359955718741, -0.33671343047171831, -0.4544564358657226, 0.57318379171192646, -0.27865237009245902, -0.051289462600834668, 0.83679822669364512, 0.12434406753163785, -0.39840185642242432, -0.34280591818969697, -0.35884603817248717, 0.47201457069604658, 0.093414641334675252, -0.027845924254506826, 0.14642265066504478, 0.35153094492852688]

People 10: 
Tresholds = 0.43058722497895363
Centroid = [-0.13240971835330129, -0.054696993785910308, 0.12116098770638928, 0.48884340585209429, -0.4539739356841892, -0.059829823207110167, -0.18436269466474187, 0.10980193130671978, -0.60497783427126706, 0.55241250386461616, -0.81354571250267327, 0.12129143439233303, -0.1114782493095845, 0.89583512861281633, 0.25620259577408433, -0.45955272321589291, -0.63094999652821571, -0.60107168927788734, -0.60983584588393569, 1.2218767516314983, 0.0091405024286359549, 0.19095209607621655, 0.65525418892502785, 0.1023146131192334, 0.2235014681937173, 0.12170115977642126, -0.03623638694989495, -0.38722805865108967, -0.071040541166439652, -0.77043516840785742, 0.29630354819528293, -0.29376247862819582, -0.13644521238165908, 0.14318287407513708, 0.20547813299344853, -0.16244728246238083, -0.93321561254560947, -0.58986707171425223, 0.86310620419681072, -0.36183210462331772, 0.001882983255200088, -0.080868581193499267, 0.20015740767121315, 0.39770646661054343, -0.83832045225426555, 0.0047950264415703714, -0.52659431740175933, -0.023215167340822518, -0.24711586779449135, -0.34033583611017093, -0.42880838096607476, 0.57683045859448612, -0.28437866398598999, -0.032811074052006006, 0.79251244850456715, 0.13663194363471121, -0.42521611135452986, -0.30006616504397243, -0.344466355338227, 0.46235041200998239, 0.083336271927691996, -0.061683112755417824, 0.14376921602524817, 0.32357290189247578]

People 11: 
Tresholds = 0.4053325983812101
Centroid = [-0.098576408810913563, -0.046939714229665697, 0.18276517919730395, 0.50946768699213862, -0.44234882108867168, -0.077002942329272628, -0.20403930643806234, 0.13239318685373291, -0.5843342503067106, 0.55154836073052138, -0.81963713793084025, 0.14842630212660879, -0.079394967760890722, 0.88218269869685173, 0.24396194447763264, -0.4622738235630095, -0.63627278385683894, -0.59793008025735617, -0.60863963124575093, 1.2358084525913, 0.0062133495812304318, 0.23447406437480822, 0.67628850997425616, 0.075742616259958595, 0.21742248465307057, 0.14114438231626991, -0.021239306108327582, -0.40320915507618338, -0.077660207636654377, -0.76384215522557497, 0.28338511660695076, -0.28442757937591523, -0.13439275650307536, 0.18332551512867212, 0.21078000994748436, -0.20398556289728731, -0.95024334453046322, -0.59687268457491882, 0.83994321152567863, -0.37353257110225968, -0.0043659006478264928, -0.094714866718277335, 0.19830094487406313, 0.40741301595699042, -0.84026639210060239, 0.0027486042963573709, -0.50204360799398273, -0.021412648959085345, -0.25768893014173955, -0.35986507184861694, -0.43925660313107073, 0.57430869340896606, -0.26754002529196441, -0.036985505721531808, 0.81408743280917406, 0.1283572397660464, -0.43250272795557976, -0.28286865865811706, -0.34235542261740193, 0.46037021046504378, 0.092291026638122275, -0.063132567564025521, 0.12760527525097132, 0.31770986481569707]

People 12: 
Tresholds = 0.41814461406785997
Centroid = [-0.13261095457710326, -0.019896651967428625, 0.13544911460485309, 0.50746678793802857, -0.48805255000479519, -0.099834908964112401, -0.18082614804734476, 0.13529814354842529, -0.57250928692519665, 0.59778506960719824, -0.83823862299323082, 0.12519529170822352, -0.12900144164450467, 0.86879653669893742, 0.25726265273988247, -0.47252073581330478, -0.65112546365708113, -0.59505963232368231, -0.59703816007822752, 1.2522250022739172, 0.052422443230170757, 0.22545562061714008, 0.69720819499343634, 0.11941997130634263, 0.22463398345280439, 0.13860071578528732, -0.025676999532151967, -0.44434713723603636, -0.075559594202786684, -0.79248448088765144, 0.29623878588608932, -0.27765043766703457, -0.14077196264406666, 0.21257592644542456, 0.20714810217032209, -0.21641118288971484, -0.98261137772351503, -0.62779338005930185, 0.8521561985835433, -0.37924386380473152, -0.033444356522522867, -0.079324866295792162, 0.23402465810067952, 0.43740097503177822, -0.85537180677056313, -0.0028811362572014332, -0.5174671458080411, 0.004105642787180841, -0.24823292618384585, -0.36168760976579506, -0.46774942521005869, 0.58286128845065832, -0.29094337637070566, -0.044493904220871627, 0.85219192504882812, 0.14584621612448245, -0.41234801337122917, -0.33490044064819813, -0.39631174871465191, 0.45239329256583005, 0.099218323390232399, -0.042700662277638912, 0.14783129864372313, 0.35250639240257442]

People 13: 
Tresholds = 0.40130609869491307
Centroid = [-0.11439587408676744, -0.03232531005050987, 0.14514507167041302, 0.51679848151979968, -0.46219735569320619, -0.067776366486214101, -0.20452437904896215, 0.13600290747126564, -0.578536715824157, 0.57220193836838007, -0.82810060912743211, 0.13980737747624516, -0.079699869500473142, 0.87744020484387875, 0.24864818586502224, -0.46686790836974978, -0.62844466045498848, -0.58461553050437942, -0.63050516741350293, 1.2154458686709404, 0.025062055210582912, 0.22476334759267047, 0.68285019416362047, 0.089484575553797185, 0.2327100356342271, 0.11401223111897707, -0.038167714286828414, -0.40180044143926352, -0.075668743811547756, -0.75857615005224943, 0.27832256544206757, -0.30409340187907219, -0.12707760016201064, 0.1721973706735298, 0.19538965442916378, -0.17312220088206232, -0.94648648984730244, -0.57100236864062026, 0.8444348203483969, -0.39408774950425141, -0.0050023214425891638, -0.082096494734287262, 0.20281829836312681, 0.41649136028718203, -0.83782679215073586, 0.038567448398680426, -0.49851748300716281, -0.031009493861347437, -0.24837427883176133, -0.38288253202335909, -0.42694257432594895, 0.57156927359756082, -0.26640240161214024, -0.033798110438510776, 0.81977123697288334, 0.13590651343110949, -0.42109066527336836, -0.31256066041532904, -0.34885257558198646, 0.4577185022062622, 0.1027467506355606, -0.063254684442654252, 0.15046277106739581, 0.31175151735078543]

People 14: 
Tresholds = 0.4027706454950385
Centroid = [-0.12506753276102245, -0.033662634319625795, 0.1553693461464718, 0.51198493270203471, -0.49593745381571352, -0.081923988880589604, -0.21345561186899431, 0.16516995808342472, -0.59057073947042227, 0.59115368581842631, -0.85053131077438593, 0.12382504646666348, -0.1037872217129916, 0.86623896891251206, 0.2624120555119589, -0.49300507735460997, -0.64576074527576566, -0.58459199033677578, -0.61603452044073492, 1.2714520134031773, 0.034388369938824326, 0.23845880798762664, 0.68902806844562292, 0.11061521217925474, 0.22053943574428558, 0.13592717683059163, -0.013552467920817435, -0.41175339638721198, -0.080948837916366756, -0.78051340486854315, 0.29658486814878415, -0.27587532054167241, -0.15220389049500227, 0.18085882184095681, 0.22003448288887739, -0.21578417287673801, -0.94308477733284235, -0.6193342455662787, 0.88396664056926966, -0.41869331151247025, -0.013864309876225889, -0.12087157531641424, 0.22875758068403229, 0.43463903467636555, -0.86734611727297306, 0.019391162204556167, -0.52037773700430989, -0.0013758823042735457, -0.27946296142181382, -0.33817707467824221, -0.4263359671458602, 0.6188378157094121, -0.28238395717926323, -0.042330658063292503, 0.83518650848418474, 0.134264359716326, -0.38570661190897226, -0.35624623531475663, -0.3726183264516294, 0.46377671786467545, 0.10895252466434613, -0.081706680124625564, 0.18491814844310284, 0.33082846808247268]

People 15: 
Tresholds = 0.40265464316820726
Centroid = [-0.12633060477674007, -0.012814863934181631, 0.15523316879989579, 0.49246084725018591, -0.44968423550017178, -0.092330501182004809, -0.17771466201520525, 0.12697524332907051, -0.56857648189179599, 0.59294868679717183, -0.85464677144773304, 0.1349871241254732, -0.11382211116142571, 0.89207578636705875, 0.25904965749941766, -0.46531118405982852, -0.58541472454089671, -0.57913305424153805, -0.57907907175831497, 1.227679779753089, 0.04146990244043991, 0.225636416755151, 0.66108047356829047, 0.11624128866242245, 0.25509719306137413, 0.12053177421330474, -0.046559928334318101, -0.41222444921731949, -0.090836013667285442, -0.76030913600698113, 0.31306739035062492, -0.28593348024878651, -0.14664498902857304, 0.18138648744206876, 0.17983621696475893, -0.17694279947318137, -0.9595336290076375, -0.61392123345285654, 0.8389285677112639, -0.37205230770632625, -0.01646150927990675, -0.071894798777066171, 0.20691012247698382, 0.40027104190085083, -0.8338114945217967, 0.016062727372627705, -0.47776337142568082, -0.017018741345964372, -0.24357695103390142, -0.377937888566521, -0.43780517741106451, 0.56194635364226997, -0.24504799395799637, -0.0563795535126701, 0.78226770367473364, 0.16257313138339669, -0.42622316814959049, -0.31773482938297093, -0.35595678631216288, 0.4342886520025786, 0.072351782640907913, -0.072139310417696834, 0.16581218736246228, 0.28580657939892262]

People 16: 
Tresholds = 0.3830716632038821
Centroid = [-0.10239338187966496, -0.027328675147145987, 0.14897812774870545, 0.52139826025813818, -0.44666546653024852, -0.084687622962519526, -0.17203372044605203, 0.12671759660588577, -0.57120180130004883, 0.53832581057213247, -0.82457672338932753, 0.14636610681191087, -0.098379688570275903, 0.86425569653511047, 0.29261678818147629, -0.43465345213189721, -0.58611833595205098, -0.57542474707588553, -0.62209065409842879, 1.1954690217971802, 0.020792721072211862, 0.19861106475582346, 0.6350268954411149, 0.10805409029126167, 0.20871937123592943, 0.12751620437484235, -0.012680614716373384, -0.37758036213926971, -0.065412733471021056, -0.76388882007449865, 0.31358456239104271, -0.25380608963314444, -0.14934740751050413, 0.16878513887058944, 0.17979086830746382, -0.13703433750197291, -0.92239319207146764, -0.60925784943538019, 0.86012265458703041, -0.37964278995059431, -0.014499615295790136, -0.062342613702639937, 0.19845199305564165, 0.38724719500169158, -0.82733143866062164, 0.035953098849859089, -0.48136257380247116, -0.020257152616977692, -0.22840507765067741, -0.33751453965669498, -0.45274233492091298, 0.57283856789581478, -0.29449562926311046, -0.022509938338771462, 0.76337252388475463, 0.14796507987193763, -0.45336249377578497, -0.28718220081645995, -0.32620956189930439, 0.42389549931976944, 0.026311416062526405, -0.056728122988715768, 0.15294395945966244, 0.32323355111293495]
* Send the github link by mail
