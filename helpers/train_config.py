NEpochs_max = 150

trainingParameters = {
	"Lambda": [5.] * (NEpochs_max), 								#Punishing term for gradient blow-up for the critic.
	"kappa_e": [0.01] * (NEpochs_max), 							#Punishing term for energy regression, default: 0.01
	"kappa_p": [0.01] * (NEpochs_max), 							#Punishing term for position regression, default: 0.01
	"DoCTrain": [True] * NEpochs_max, 							#Training of critic?
	"DoEnergyRecoTrain": [True] * 50 + [False] * 100, 					#Training of E?
	"DoPositionRecoTrain": [True] * 50 + [False] * 100,  				#Training of P?
	"DoGTrain": [True] * (NEpochs_max), 							#Training of G?
	"trainGTotEvery": [10]  * (NEpochs_max), 						#Train Gtot every N iterations through the batches
	"learning_rateC": [0.0005]*60 + [0.0002]*20 + [0.0001]*20 + [0.00005]*50, 									#Learning rate of for adam for the critic
	"learning_rateD": [0.0005]*NEpochs_max, 									#Learning rate of for adam for discriminator
	"learning_rateG": [0.001]*70 + [0.0005]*20 + [0.0002]*10 + [0.0001]*50, 									#Learning rate of for adam for G 
	"beta1": 0.0, 												#Beta1 for adam for G/D 
	"beta2": 0.9, 												#Beta2 for adam for G/D 
	"learning_rateE": [0.00005]*NEpochs_max, 									#Learning rate of for adam for E 
	"learning_rateP": [0.00005]*NEpochs_max									#Learning rate of for adam for P 
}
