# Uczenie ze wzmocnieniem
W ramach projektu zostało stworzone trzech agentów-przeciwników grających w Connect3 - pomniejszoną wersję gry Connect4.

## Warianty agentów:
- uczenie pasywne - algorytm value iteration
- uczenie aktywne - algorytm Q-learning
- przybliżenie funkcji wartości funkcją liniową

## Uczenie pasywne 
W pliku "passive learning agent - function generation.ipynb" możliwe jest wygenerowanie funkcji wartości dla gry jako plik pickle. Natomiast w pliku "passive_connect3_terminal.py" możliwe jest załadowanie wyliczonej funkcji wartości i zagranie w grę Connect3 przeciwko nauczonemu agentowi.

## Uczenie aktywne
W pliku "connect_3_active_learning_terminal.py" możliwe jest zagranie przeciwko agentowi nauczonemu przez algorytm Q-learning. Po uruchomieniu pliku następuje uczenie agenta, które trwa około 10 minut. Po nauce można zagrać przeciwko agentowi.

## Value approximation
W pliku 'Value approximation agent.ipynb' znajduje się implementacja agenta przybliżającego funkcję wartości funkcją liniową wraz z wyliczonymi wagami. W pliku możliwe jest zagranie przeciwko agentowi w grę.
