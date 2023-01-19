import { useEffect, useState } from "react";
import CurrencySelector from "../currencySelector/CurrencySelector";
import MoneyInput from "../MoneyImput/MoneyImput";

const currency  = [
  {
    id: 1,
    name: 'USD',
    avatar:'https://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_the_United_States.svg',
  },
  {
    id: 2,
    name: 'CHF',
    avatar:'https://cdn.britannica.com/43/4543-004-C0D5C6F4/Flag-Switzerland.jpg',
  },
  {
    id: 3,
    name: 'EUR',
    avatar:'https://m.media-amazon.com/images/I/614JLqsvMoL._AC_SL1500_.jpg',
  },
]

export default function ChangeCard() {


  const [currencyOption1, setCurrencyOption1] = useState(currency[0]);
  const [currencyOption2, setCurrencyOption2] = useState(currency[0]);
  const [value1, setValue1] = useState(0);
  const [value2, setValue2] = useState(0);

  useEffect(()=>{
    changeCalculation1(value1);
  }, [currencyOption1]);

  useEffect(()=>{
    changeCalculation2(value2);
  }, [currencyOption2]);

  const selectCurrency1 = cur => {
    setCurrencyOption1(cur);
    changeCalculation1(value1);
  }

  const selectCurrency2 = cur => {
    setCurrencyOption2(cur);
    changeCalculation2(value2);

  };

  const changeCalculation1 = value => {
    if(currencyOption1 === currencyOption2){
      setValue2(value);
      setValue1(value);
    } else {
      setValue2(value*2);
      setValue1(value);
    }
  }

  const changeCalculation2 = value => {
    if(currencyOption1.name === currencyOption2.name){
      setValue2(value);
      setValue1(value);
    } else {
      setValue1(value/2);
      setValue2(value);
    }
  }

  const swapFC = () => {
    const temp1 = currencyOption2;
    const temp2 = currencyOption1;
    setCurrencyOption1(temp1);
    setCurrencyOption2(temp2);
  }

    return (
      <><div className="sm:py-20 pb-44 px-2 py-2 sm:px-48 justify-center flex flex-col h-[60vh]  pt-40 mt-40 mb-20 ">
        <h2 className="mt-4 font-bold  text-2xl text-black ">Taux de change du jour</h2>
        <h3 className="mt-4 text-xl leading-8 text-gray-800 sm:text-left sm:w-1/2 mb-4">GMT Change, leader sur le marché du change à Genève, avec certainement un des meilleurs taux de change à la vente et à l’achat</h3>
        <h3 className="mt-4 text-xl leading-8 text-gray-800 sm:text-left sm:w-1/2 mb-4">Ce taux est susceptible d’être modifié à tout moment.</h3>
        <div class="bg-white p-6 rounded-lg sm:py-24 shadow-lg  text-center	">
          <h2 class="text-2xl font-bold mb-2 text-gray-800">Convert {currencyOption1.name} to {currencyOption2.name}</h2>
          <div class="flex sm:items-end justify-center sm:flex-row flex-col ">
            <div className="sm:w-1/4 sm:pb-0 pb-4 ">
              <MoneyInput onChange={e => changeCalculation1(e.target.value)} value={value1}></MoneyInput>
            </div>
            <div className="sm:pl-8 sm:pb-0 pb-4">
              <CurrencySelector selectCurrency={selectCurrency1} swap={currencyOption1}></CurrencySelector>
            </div>
            <button onClick={swapFC} className="inline-flex items-center self-center justify-center w-10 h-10 mx-2 text-pink-100 transition-colors duration-150 bg-blue-600 rounded-full focus:shadow-outline hover:bg-blue-800">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
              </svg>

            </button>
            <div className="sm:pr-8 sm:pt-0 pt-4">
              <CurrencySelector selectCurrency={selectCurrency2} swap={currencyOption2}></CurrencySelector>
            </div>
            <div className="sm:w-1/4 sm:pt-0 pt-4">
              <MoneyInput value={value2} onChange={e => changeCalculation2(e.target.value)}></MoneyInput>
            </div>
          </div>
        </div>
      </div>
      
{/*      <div class="sm:h-[40vh]  justify-start flex flex-col w-screen shadow-lg  sm:text-center pt-32 sm:pt-12 sm:mt-0">
        <h2 className="mt-4 font-bold  text-2xl text-black ">Compare our rates</h2>
        <h3 className=" text-xl leading-8 text-gray-800 sm:text-center">Compare our Euro exchange rate today to see how many Euro 500 CHF will buy you</h3>
          <div className="flex flex-col text-center sm:flex-row space-between w-screen content-center justify-items-center justify-center gap-x-44 gap-y-4 pt-4">
            <div  className="text-blue-800 font-semibold">
              <h2>$550</h2>
              <h3>GMT Change</h3>
            </div>
            <div className="text-gray-600">
              <h2>$500</h2>
              <h3>Migros Change</h3>
            </div>
            <div className="text-gray-600">
              <h2>$498</h2>
              <h3>Change X</h3>
            </div>
            <div className="text-gray-600">
              <h2>$502</h2>
              <h3>Revolut</h3>
            </div>
          </div>
    </div> */}
      </>

    )
  }