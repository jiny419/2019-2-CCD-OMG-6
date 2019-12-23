(function () {
    var video = document.getElementById('video'),
        vendorUrl = window.URL || window.webkitURL;
        canvas = document.getElementById('canvas'),
        context = canvas.getContext('2d'),
        photo = document.getElementById('photo'),
        image = document.getElementById('image')

    navigator.getMedia = navigator.getUserMedia ||
        navegator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;

    // capture Video
    navigator.getMedia({
        video: true,
        audio: false
    }, function (stream) {
        video.src = vendorUrl.createObjectURL(stream);
        video.play();
    }, function (error) {

    })
    document.getElementById('capture').addEventListener('click', function () {
        context.drawImage(video, 0, 0, 256, 256);
        // console.log(canvas.toDataURL('image/png'));

        photo.setAttribute('src', canvas.toDataURL('image/png'));
        image.setAttribute('value', canvas.toDataURL('image/png'));
 
    });




})();