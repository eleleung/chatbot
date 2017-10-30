// CITS4404 Group C1
// serves the UI component
// code from: https://github.com/suriyadeepan/easy_seq2seq/

var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
    $messages.mCustomScrollbar();
    setTimeout(function() {
    }, 100);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate(){
    d = new Date();
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    updateScrollbar();
    interact(msg);
    setTimeout(function() {

    }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})


function interact(message){
    $('<div class="message loading new"><figure class="avatar"><img src="/static/img/red-nao-robot.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
        $.post('/message', {
        msg: message,
    }).done(function(reply) {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="/static/img/red-nao-robot.png" /></figure>' + reply['text'] + '</div>').appendTo($('.mCSB_container')).addClass('new');

        setDate();
        updateScrollbar();
    }).fail(function() {
        alert('Error with function');
    });
}
