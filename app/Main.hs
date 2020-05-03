module Main where

import Data.Matrix
import Lib

getOr :: (Matrix Double, Matrix Double)
getOr = (xs, ys)
  where
    xs = transpose $ fromLists [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = fromLists [map (foldl numOr 0) $ toLists $ transpose xs]

numOr :: (Eq a1, Eq a2, Num a1, Num a2, Num p) => a1 -> a2 -> p
numOr a b =
  if a /= 0 || b /= 0
    then 1
    else 0

main :: IO ()
main = do
  putStrLn "Test is done using function `or` with following data:"
  let xss = fst getOr
  let ys = snd getOr
  putStrLn "X:"
  putStrLn $ prettyMatrix xss
  putStrLn "Y:"
  putStrLn $ prettyMatrix ys
  let iterations = 1000
  let rate = 0.05
  let (predictions, _, ws, b) = model xss ys xss ys iterations rate
  putStrLn $
    "Predictions using " ++
    show iterations ++ " iterations with learning rate " ++ show rate ++ ":"
  putStrLn $ prettyMatrix predictions
  putStrLn "Weights:"
  putStrLn $ prettyMatrix ws
  putStrLn $ "Bias: " ++ show b
