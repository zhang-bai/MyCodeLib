from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.header import Header
import smtplib

import matplotlib.pyplot as plt
import numpy as np


def mails(text='None', title='test', fig_add=None):
    # qq邮箱smtp服务器
    # xxxxxxxxxxxxxxx改为相关邮箱信息
    host_server = 'smtp.qq.com'
    pwd = 'xxxxxxxxxxxxxxxxx'

    sender = 'xxxxxxxxxxxxxx@qq.com'
    receiver = 'xxxxxxxxxxx@mails.jlu.edu.cn'

    msg = MIMEMultipart('alternative')
    msg["Subject"] = Header(title,"utf-8")
    msg["From"] = sender
    msg["To"] = receiver

    if fig_add :
        print('sending picture')
        fp = open(fig_add, 'rb')
        images = MIMEImage(fp.read())
        fp.close()
        fig_name = fig_add.split('.')[-2]
        fig_name = fig_name.split('/')[-1] + '")'
        images["Content-Disposition"] = 'attachment; filename="'+fig_name
        images.add_header('Content-ID', '<image1>')
        msg.attach(images)
        msg.attach(MIMEText("<p><img src='cid:image1' type='image/svg+jpg+xml' style='width:100%'></p>",'html','utf-8'))
    msg.attach(MIMEText(text,"html","utf-8"))

    # sending
    smtp = smtplib.SMTP()
    smtp.connect(host_server)
    smtp.login(sender, pwd)
    print('Sending')
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
    print('OK')


def draw_(epoch, acc=np.array([]), loss=np.array([]), title='test',):
    epoches = range(epoch)

    if acc.size and loss.size:
        assert 0 ,'test'
        plt.plot(epoches)
    elif acc.size:
        # row,col = acc.shape
        # row is the number of epoches and col in name_list
        name_list = ['train acc','valid_acc','test_acc']
        nb = acc.shape[-1]

        for i in range(nb):
            plt.plot(epoches,acc[:,i],'-',label=name_list[i])
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=nb, mode="expand", borderaxespad=0.)
        plt.legend()

        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(title)
        plt.tick_params(direction='in')

        fig_name = './acc_'+str(epoch)+'.jpg'
        fig = plt.gcf()
        fig.savefig(fig_name,dpi=600)

        add=fig_name

    elif loss.size:
        assert 0,'test loss'
        plt.plot(epoches)
    else:
        assert 0,"No acc and loss"
        
    return add


def send_report(epoch,acc=np.array([]),loss=np.array([]),message='',title='test'):
    title = title + " |epoch --%d" % (epoch)
    add = draw_(epoch,acc,loss,title)
    mails(message,title=title,fig_add=add)


if __name__ == "__main__":
    epoch = 10
    acc = np.random.rand(10,3)
    add = draw_(epoch,acc=acc)
    send_report(epoch,acc,message='测试文档')
