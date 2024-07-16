
# TinyIDS - An IoT Intrusion Detection System by Tiny Machine Learning

The proliferation of Internet-of-Things (IoT) devices across different sectors, including healthcare, automotive, and industrial automation, has elevated the potential for cyberattacks on critical assets. While machine learning technology can be employed to identify malicious behaviors, it is not readily applicable to Internet-of-Things (IoT) devices due to the necessity of energy-intensive and expensive devices that are not easily deployable in such contexts. Furthermore, privacy and security policies, in addition to latency constraints, can impose limitation on the transmission of sensitive data to remote servers. To address this challenge, the emerging field of TinyML presents a potential solution for the direct implementation of machine learning algorithms on resource-constrained devices. This paper presents an implementation of a TinyIDS, which employs Tiny Machine Learning techniques to detect attacks on sensor networks and malicious behavior of compromised intelligent objects. Train-on-board is utilized to train and analyze local data without transferring sensitive data to remote or non-trustworthy cloud services. The solution was tested on common MCU-based devices and ToN_IoT dataset. 


## Documentation

- Benchmark: this folder contains the benchmark code developed to work with Arduino UNO R3 and INA219 sensor. It allows you to measure voltage, power and current of a second development board connected via serial communication. The workflow of this code follows the following finite state machine.

![alt text](https://raw.githubusercontent.com/iotresearchunisa/TinyIDS/main/Documnetation/img/TinyML_MSF.png)

- Dataset: this folder contains the ToN_IoT dataset used for the experiments.

- PythonExample: this folder contains a first implementation of the neural network written in Python.

- TinyIDS_Implementation: this folder contains the three implementations (arduino_h7, esp32, esp8266) of the neural network consisting of the following parameters:
    - Input: 17 * 30
    - Hidden Layer: 32 * 32
    - Output: 9
	
## Authors

- [@Pietro Fusco](https://docenti.unisa.it/064613/home)
- [@Gennaro Pio Rimoli](https://github.com/gprimoli)
- [@Massimo Ficco](https://docenti.unisa.it/058291/home)
## Mention our work
```
@InProceedings{Fusco2024,
    author="Fusco, Pietro and Rimoli, Gennaro Pio and Ficco, Massimo",
    title="TinyIDS - An IoT Intrusion Detection System by Tiny Machine Learning",
    booktitle="International Conference on Computational Science and Its Applications - ICCSA 2024 Workshops",
    year="2024",
    publisher="Springer Nature Switzerland",
}


```
## Acknowledgements

The work is part of the research activities realized within the projects Federated Learning for Generative Emulation of Advanced Persistent Threats (FLEGREA), CUP. E53D23007950001, Bando PRIN 2022, as well as SERICS (PE00000014) under the NRRP MUR program funded by the EU - NGEU.

