
from cmath import sqrt
import yfinance as yf
import pandas as pd
from datetime import date, datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from simple_term_menu import TerminalMenu
from math import log, exp
from scipy.stats import norm

def getVolatility(stock):

    
    data = stock.history(period="6mo")
    data['PCT'] = data['Close'].astype(float).pct_change().apply(lambda x: np.log(1+x))
    data_pct = data['PCT']

    mean = data_pct.mean()
    sqd = data_pct.apply(lambda x: (x-float(mean))**2)
    num = sqd.sum()
    variance = num/(data_pct.count()-1)
    
    volatility = sqrt(variance)

    return volatility*sqrt(252)

def getDaysOut(stock):
        optionsMenu = TerminalMenu(stock.options, title="Please select Exp. date:")
        optionDecision = optionsMenu.show()
        today = datetime.date(datetime.now())
        optionExp = datetime.strptime(stock.options[optionDecision], '%Y-%m-%d').date()
        out = (optionExp-today).days
        return out

def d1(S,K,T,r,q,sigma):
    return(log(S/K)+(r - q + sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,q,sigma):
    return d1(S,K,T,r,q,sigma)-sigma*sqrt(T)

def call(S,K,T,r,q,sigma):
    return S*exp(-q*T)*norm.cdf(d1(S,K,T,r,q,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,q,sigma))

def put(S,K,T,r,q,sigma):
    return K*exp(-r*T)-S*exp(-q*T)+call(S,K,T,r,q,sigma)

def main():
    while(True):

        options = ["[1] Begin", "[2] Quit"]

        terminal_menu = TerminalMenu(options)
        menu_entry_index = terminal_menu.show()
        
        if(menu_entry_index == 1):
            break

        
        try:
            ticker = str(input("Please enter a ticker: "))
            stock = yf.Ticker(ticker)


            ###### Model Goes Here ######

            #### Variables ####
            cp = ['[1] Call', '[2] Put']
            callOrPut = TerminalMenu(cp, title='Call or Put Option?')
            callOrPutOption = callOrPut.show()


            if(callOrPutOption==0):

                price = stock.history(period='1d')['Close'][0]
                r = 0
                sigma = getVolatility(stock)
                K = float(input("Please enter strike price for " + stock.ticker + ": "))
                q = 0
                T = getDaysOut(stock)/365

                S=np.linspace(price-50,K+100,20)
                h=np.maximum(S-K, 0)
                C = [call(Szero,K,T,r,q,sigma) for Szero in S]
                plt.figure()
                plt.title(stock.ticker + " CALL")
                plt.plot(S, h, 'b-.', lw=3, label='Intrinsic Value') 
                plt.plot(S, C, 'r', lw=3, label='Time Value') 
                plt.grid(True)
                plt.legend(loc=0)
                plt.xlabel('Spot Price of Underlying')
                plt.ylabel('Option Value in Current Time')
                #print(C)
                plt.savefig('graph.png')
            else:

                price = stock.history(period='1d')['Close'][0]
                r = 0
                sigma = getVolatility(stock)
                K = float(input("Please enter strike price for " + stock.ticker + ": "))
                q = 0
                T = getDaysOut(stock)/365

                S=np.linspace(price-100,K+50,20)
                h=np.maximum(S-K, 0)
                P = [put(Szero,K,T,r,q,sigma) for Szero in S]
                plt.figure()
                plt.title(stock.ticker + " PUT")
                plt.plot(S, h, 'b-.', lw=3, label='Intrinsic Value') 
                plt.plot(S, P, 'r', lw=3, label='Time Value') 
                plt.grid(True)
                plt.legend(loc=0)
                plt.xlabel('Spot Price of Underlying')
                plt.ylabel('Option Value in Current Time')
                #print(C)
                plt.savefig("Graph.png")
        except:
            print("")

if __name__ == '__main__':
    main()