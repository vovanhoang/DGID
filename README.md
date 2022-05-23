# SDAID: Towards A Hybrid Signature and Deep Analysis-based Intrusion Detection Method


Perform an ensemble approach that combines rule-based IDPS and Deep Neural networks for the intrusion detection system.

For the paper "SDAID: Towards A Hybrid Signature and Deep Analysis-based Intrusion Detection Method"!

In order to validate and evaluate the effectiveness of the GID method, we use two sub-datasets

1. The first dataset (DS1) includes CSE-CIC-IDS2018 and the dataset (more than 20,000 flows of SQL injection attacks) that we directly built through the testbed system. Labeling the dataset is also an important step that takes considerable time and effort, so we also create a tool by python for automatically labeling the dataset. The goal of using this dataset is to verify DGID's ability to detect network attacks by analyzing the network traffic correctly.

2. The second dataset (DS2): we use the classic NSL-KDD and the up-to-date CSE-CIC-IDS2018 as benchmark datasets. For our experiment, within the dataset exists, we divided it into four different classes of attacks: (1) Denial of Service (DoS), which tries to shut down the traffic flow to and from the target; this is the most common attack in the dataset; (2) Probe, this kind of attacks, tries to get information from a network the goal of this attacks is to act like a thief and steal important information; (3) User to Root (U2R), such as privilege escalation attacks, with a normal user account and tries to gain access to the system or network, as a super-user(root); Remote to Local(R2L), tries to gain access to a remote machine, an attacker do not have local access to the system/network, and tries to hack into the network. Thus, our goal when using this data set is to objectively compare the efficiency of the DGID method with other studies using the same dataset.

From these two sub-datasets, we constitute a dataset for testing. The dataset contains two parts: training and testing at the ratio of 7:3.

For full access to the source code and datasets, please download them by following the link: 
https://drive.google.com/drive/folders/122mwJiO2fHPZxSdjpZ_Zd5GlSTp7Id87?usp=sharing

With the contributions of the authors:

Hoang V. Vo - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam

Hoa N. Nguyen - Corresponding author - hoa.nguyen@vnu.edu.vn - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam

Tu N. Nguyen is with the Department of Computer Science, Kennesaw State University, Marietta, GA 30060, USA

Hanh P. Du - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam
