<!DOCTYPE html>
<html lang="zh">
<head>







    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <meta name="viewport" content="width=device-width, maximum-scale=1.0, user-scalable=0, initial-scale=1.0">
    <meta content="telephone=no" name="format-detection">

    <title>Baidu Authentication Platform</title>

    <link rel="icon" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="shortcut icon" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="bookmark" href="/ico/favicon.ico" type="image/x-icon"/>
    <link rel="stylesheet" href="/css/base.css?v=2.0">
    <link rel="stylesheet" href="/css/login.css?v=2.0">
    <link rel="stylesheet" href="/js/lib/action.min.css?v=2.0">
</head>

<body>









<div class="wrap">

    <div class="header">
        <a href="/login">
            <img class="logo" src="/images/logo.png">
        </a>

        <div class="more">
            <img src="/images/more.svg">
        </div>

        <div class="right">
            <a href='/login?service=http%3A%2F%2Fgitlab.baidu.com%2Fusers%2Fauth%2Fcas3%2Fcallback%3Furl&amp;locale=zh_CN'>
                简体中文
            </a>
        </div>
    </div>

    <div class="h5-nav hide">
        <div class="container">




                    <a href="/manage/resetpassword">
                        Forgot your password
                    </a>


            <a href='/login?service=http%3A%2F%2Fgitlab.baidu.com%2Fusers%2Fauth%2Fcas3%2Fcallback%3Furl&amp;locale=zh_CN'>
                简体中文
            </a>
            <a href="/manage/help">
                Help
            </a>
            <s>
                <i></i>
            </s>
        </div>
    </div>

    <div class="shade">
        <img src="/images/loginSuccess/wait.gif"/>
        <p>Signing...</p>
    </div>

    <div class="login">
        <div class="content">
            <div class="box">
                <div class="loading"></div>
                <div class="toast-wrap">
                    <span class="toast-msg">Network error, please try again later</span>
                </div>
                <div class="tooltip">
                    <div class="tooltip-arrow"></div>
                    <div class="tooltip-inner">
                        <div>Please make sure mobile infoflow:</div>
                        <div>IOS version higher than 6.10.0</div>
                        <div>Android version higher than 6.10.0</div>
                    </div>
                </div>
                <div class="nav">
                    <div class="h5-title">
                        Account Login
                    </div>
                    <span class="tab on" data-type="email" id="1">
                        Account Login
                    </span>
                    <span class="line">|</span>
                    <span class="tab" data-type="scan" id="2">
                        QRCode
                    </span>
                    <span class="line">|</span>
                    <span class="tab" data-type="token" id="3">
                        RSA Token
                    </span>
                </div>


                <form id="form-email" action="/login;jsessionid=881f536371be4560b9f2fd5809f60a3f" method="post">
                    <div class="email-area">


                        <div class="li list text username">
                            <input type="text" id="username" data-type="username" name="username" maxlength="90"
                                   value=''
                                   placeholder='Baidu Account'>
                        </div>
                        <div class="li list text password">
                            <input type="password" id="password-email" data-type="password"
                                   placeholder='Account Password'>
                        </div>
                        <div class="li attach">
                            <span class="checkbox check"></span>
                            <span>Remember me</span>
                        </div>

                        <div class="li hint">
                            <em></em>
                        </div>

                        <div class="li bt-login commit" id="emailLogin">
                            <span>Sign in</span>
                        </div>

                        <div class="li changeLoginType">
                            <span class="show-actions">Change Login Type</span>
                        </div>

                        <div class="li other">
                        <span class="help">



                                    <a href="/manage/resetpassword">
                                        Forgot your password
                                    </a>


                            <a href="/manage/help" target="_blank">
                                Help
                            </a>
                        </span>
                        </div>
                        <input type="hidden" name="password" id="encrypted_password_email" value=''/>
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-email" value="LT-570627608154980353-F35NS">

                        <input type="hidden" name="execution" value="e1s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='1' name="type">
                    </div>
                </form>


                <form id="form-token" action="/login;jsessionid=881f536371be4560b9f2fd5809f60a3f" method="post">
                    <div class="token-area">


                        <div class="li list text username">
                            <input type="text" id="token" data-type="username" name="username" maxlength="90"
                                   value=''
                                   placeholder='Baidu Account'>
                        </div>
                        <div class="li list text password">
                            <input type="password" id="password-token" data-type="password"
                                   placeholder='PIN+RSA(RSA Token)'>
                        </div>
                        <div class="li attach" style="display: none">
                            <span class="checkbox"></span>
                            <span>Remember me</span>
                        </div>

                        <div class="li hint">
                            <em></em>
                        </div>

                        <div class="li bt-login commit" id="tokenLogin">
                            <span>Sign in</span>
                        </div>

                        <div class="li changeLoginType">

                                <span class="show-actions">Change Login Type</span>

                        </div>

                        <div class="li other">
                        <span class="help">
                            <a href="/manage/help" target="_blank">
                                Help
                            </a>
                        </span>
                        </div>
                        <input type="hidden" name="password" id="encrypted_password_token" value=''/>
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-token" value="LT-570627608154980353-F35NS">

                        <input type="hidden" name="execution" value="e1s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='3' name="type">
                    </div>
                </form>


                <form id="formQRCode" action="/login;jsessionid=881f536371be4560b9f2fd5809f60a3f" method="post">
                    <div class="qcode-area">
                        <div class="qcode" id="qcode">
                        </div>
                        <div class="scan-success">
                        </div>
                        <div class="li hint">
                            <em></em>
                        </div>
                        <div class="li changeLoginType">
                            <span class="show-actions">Change Login Type</span>
                        </div>
                        <input type="hidden" name="username" maxlength="90" id="qrCodeUsername">
                        <input type="hidden" name="password" id="qrCodePassword">
                        <input type="hidden" name="rememberMe" value="on">
                        <input type="hidden" name="lt" id="lt-qrCode" value="LT-570627608154980353-F35NS">

                        <input type="hidden" name="execution" value="e1s1">
                        <input type="hidden" name="_eventId" value="submit">
                        <input type="hidden" value='2' name="type">
                    </div>
                </form>
            </div>
        </div>
    </div>

</div>

<script src="/js/lib/flex.min.js?v=2.0"></script>
<script type="text/javascript" src="/js/lib/jquery3.2.1.min.js"></script>
<script type="text/javascript" src="/js/lib/jquery.placeholder.min.js"></script>
<script type="text/javascript" src="/js/jsencrypt.min.js"></script>
<script type="text/javascript" src="/js/lib/actions.min.js?v=2.0"></script>
<script type="text/javascript" src="/js/login.js?v=6.0"></script>
<script type="text/javascript" src="/js/header.js?v=2.0"></script>
<script type="text/javascript"
        src="/beep-sdk.js?language=en&v=1613977823535"></script>


<script type="text/javascript">
    var notnull = 'You cannot leave this field blank!',
        sp_noemail = 'Account does not include Mail suffixes, such as @baidu.com',
        sp_username = 'Baidu Account',
        sp_passwd = 'Account Password',
        sp_hardToken = 'PIN+RSA(RSA Token)',
        usernameformaterror = 'Wrong user name format!',
        usernameprompt = 'Baidu Account',
        lastLoginType = 1,
        securityLevel = 1,
        rsaPublicKey = 'MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDSzTSkeLSG1wAOAMRh4L4O78jP4KgSwvMWSnpiWUrOpGknhHMMeoESI94NXdp9DZkptocfuo6dygUOsM+YM60+EVpRg2e9yWApvj88n88+yqQSJeCTRMRS2CDKZrOqf3WOQx7X72Ogj+yTx7mE+Ld+hhrl1ghPxCulQyOnMDSzbwIDAQAB',
        beepQrCodeToken = '6eef72577bf7820f72f71e6ac90d0461f1450bf99014af3c2cacaef55b461410',
        mailBoxLoginTabName = 'Account Login',
        qrCodeLoginTabName = 'QRCode',
        mobileHiLoginTabName = 'Mobile Infoflow Login',
        hardTokenLoginTabName = 'RSA Token',
        cancelButtonName = 'Cancel';
</script>

</body>
</html>
