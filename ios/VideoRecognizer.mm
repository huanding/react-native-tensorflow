#import "VideoRecognizer.h"
#import "URLHelper.h"
#import "ImageProcessor.h"

#import <ImageIO/ImageIO.h>
#import <AVFoundation/AVFoundation.h>

#include <fstream>
#include <string>
#include <vector>

@implementation VideoRecognizer
{
    ImageProcessor * imageProcessor;
}

- (id) initWithData:(NSString *)modelUri labels:(NSString *)labelUri
{
    self = [super init];
    if (self != nil) {
    	imageProcessor = [[ImageProcessor alloc] initWithData:modelUri labels:labelUri];
    }
    return self;
}

- (NSArray *) recognize:(NSString *)videoUri maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    NSURL * url = [URLHelper toURL:videoUri];
    AVURLAsset * asset = [[AVURLAsset alloc] initWithURL:url options:nil];
    
    CGImagePropertyOrientation orientation = getOrientation(asset);

    CMTime cmTime = CMTimeMakeWithSeconds(0, 60);
    NSArray * timestamps = [NSArray arrayWithObjects: [NSValue valueWithCMTime:cmTime], nil];
    NSArray * results = [self recognizeImage:asset timestamps:timestamps orientation:orientation
        maxResults:maxResults threshold:threshold];
    return results;
}

- (NSArray *) recognizeImage:(AVAsset *)asset timestamps:(NSArray *)timestamps
                             orientation:(CGImagePropertyOrientation)orientation
                             maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    AVAssetImageGenerator * gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
    gen.appliesPreferredTrackTransform = YES;
    
    NSMutableArray * results = [[NSMutableArray alloc] init];
    
    // TODO: generateCGImagesAsynchronously for multiple thumbnail generation
    for (NSValue * timestamp in timestamps) {
        CMTime expectedTime = [timestamp CMTimeValue];
        CMTime actualTime;
        NSError * error = nil;
        CGImageRef image = [gen copyCGImageAtTime:expectedTime actualTime:&actualTime error:&error];
        
        NSArray * result = [imageProcessor recognize:image orientation:orientation maxResults:maxResults threshold:threshold];
        [results addObject:result];
    }
    
    return results;
}

CGImagePropertyOrientation getOrientation(AVAsset * asset)
{
    AVAssetTrack * videoTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] objectAtIndex:0];
    CGSize size = [videoTrack naturalSize];
    CGAffineTransform txf = [videoTrack preferredTransform];
    
    if (size.width == txf.tx && size.height == txf.ty)
        return kCGImagePropertyOrientationRight;
    else if (txf.tx == 0 && txf.ty == 0)
        return kCGImagePropertyOrientationLeft;
    else if (txf.tx == 0 && txf.ty == size.width)
        return kCGImagePropertyOrientationDown;
    else
        return kCGImagePropertyOrientationUp;
}

@end
