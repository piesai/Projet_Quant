# Projet_Quant
Testing things about quant finance.

#Projet_Pricing
After reading John C Hull's "Options,Futures and other Derivatives" i wanted to see if theory meets practice using real world finance data. This is why in this first project, i try different pricing methods for options (BSM, Monte Carlo, Binomial Trees).
        #NAIVE MODELS :
        #model_BSM : 
        Here i try to implement BSM's model directly for pricing calls on the 'APPL' stock. I implemented this model first because of its simplicty. The model gives
        really good values for inTheMoney options as we can see on the graphs. However as soon as the options are out the money, the model fails. 
        To me, this is because of two problems : 
            -This model is used for european options only. However, here, i am using it for american options and european options.
            -This model assumes volatility stays constant.
        Therefore, i hope i get better results using binomial trees ! 

        #model_Binomial_Tree : 
        After getting roughly the same results than BSM Model, i try to add dividends. This does not change anything. For the two models, i observe the same
        phenomenon : 
            -Very in the money ( K << S) : priced really well and volatility skew very high. Indeed, these movements of high loss can happen on real markets. So these
            options are priced pretty well since their volatility can be modeled.
            -Very out of the money (K >> S) : priced really poorly (cf graphs). My models predict these option to be nearly 0 whereas in real life they are a bit more.
            That is because traders associate a prime to the risk for these options. Indeed if they didn't cost anything, it would be easy money.

        #Update : 
        After changing the price of the option from "lastPrice" to "ask", i get much better results on the predictions with "implied volatility". The results are, in average, better for 'model_BSM' however they start getting bad when the options get out the money. Whereas for the binomial tree, the error stays constant with K growing. 
        These models worked very well but just for implied volatility. In real life, to establish long term strategies, it is mandatory to be able to predict options price's without having the implied volatility. Therefore, we have to implement a volatility model that recreates the volatility skew.

       