#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>

@interface ImageProcessor: NSObject
- (id) initWithData:(NSString *)modelUri labels:(NSString *)labelUri;
- (void) reset;
- (NSArray *) recognize:(CGImageRef)imageRef orientation:(CGImagePropertyOrientation)orientation
             maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold;
@end
