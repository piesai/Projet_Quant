# Projet_Quant
Testing things about quant finance.

#Projet_Pricing
After reading John C Hull's "Options,Futures and other Derivatives" i wanted to see if theory meets practice using real world finance data. This is why in this first project, i try different pricing methods for options (BSM, Monte Carlo, Binomial Trees).
        #model_BSM : 
        Here i try to implement BSM's model directly for pricing calls on the 'APPL' stock. I implemented this model first because of its simplicty. The model gives
        really good values for inTheMoney options as we can see on the graphs. However as soon as the options are out the money, the model fails. 
        To me, this is because of two problems : 
            -This model is used for european options only. However, here, i am using it for american options and european options.
            -This model assumes volatility stays constant.
        Therefore, i hope i get better results using binomial trees ! 

        #model_Binomial_Tree : 
